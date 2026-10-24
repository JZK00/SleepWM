"""SleepWM components shared by supervised training and staged checkpoint assembly."""
from __future__ import annotations

import copy
import torch
from torch import nn

from .builders import build_mainline_model
from .features import FEATURE_NAMES, FEATURE_GROUPS
from .state import RecursiveBeliefCarryCorrectWorldModel
from .readouts import (
    DirectObservationBranch, TrajectoryOutcomeAdapter, LatentHazardSafeAdapter,
    DynamicEventHazardAdapter, ReliabilityGatedEventCorrection, observation_age,
)


def default_config(compact: bool = False) -> dict:
    """Model construction only; compact reduces dimensions for software examples."""
    width, hidden = (32,32) if compact else (128,128)
    return {
        'data': {'modalities':['EEG','ECG','EMG'], 'stored_modalities':['EEG','ECG','EMG'],
                 'sample_rate':128, 'epoch_seconds':30, 'num_classes':5,
                 'history_epochs':20, 'future_horizons':[1,2,4]},
        'model': {'patch_samples':128 if compact else 64, 'd_model':width,
                  'layers':1 if compact else 4, 'heads':4, 'dropout':0.1,
                  'transition_layers':1 if compact else 2, 'transition_heads':4},
        'physiology': {'feature_names':list(FEATURE_NAMES),
                       'feature_groups':{k:list(v) for k,v in FEATURE_GROUPS.items()},
                       'hidden_dim':hidden, 'physiology_dynamics_dim':32 if compact else 64,
                       'physiology_dynamics_layers':1, 'physiology_dynamics_heads':4},
        'waveform': {'max_seconds':10,'patch_samples':64,'decoder_dim':32 if compact else 64,
                     'decoder_layers':1,'decoder_heads':4,'structured_event_heads':True,
                     'probabilistic_event_heads':True,'ecg_refractory_event_head':True,
                     'ecg_rr_bins':48,'output_baseline':'none','physiological_event_renderer':True,
                     'missing_modality_conditioning':True,'missing_physiology_calibration':True,
                     'missing_emg_teacher_calibration':True},
        'observation_repair': {'hidden_dim':hidden,'task_hidden_dim':hidden},
        'recursive': {'direct_horizons':[1,2,4],'anchor_direct':True,'hidden_dim':hidden},
        'partial_observation': {'hidden_dim':hidden,'initial_freshness_decay':0.35,
                                'initial_uncertainty_age_scale':0.15,'initial_uncertainty_horizon_scale':0.05},
        'belief_filter': {'hidden_dim':hidden,'maximum_delta':0.5,'use_dynamics':True,'correction_mode':'learned'},
        'outcome_adapter': {'hidden_dim':hidden,'maximum_stage_delta':2.,'maximum_physiology_delta':1.},
        'direct': {'feature_dim':16 if compact else 32,'hidden_dim':32 if compact else 96,'dropout':0.1},
        'fusion': {'hidden_dim':hidden,'maximum_physiology_residual':0.35,'initial_direct_gate':0.65,'dropout':0.1},
    }


class SleepWM(nn.Module):
    """State filter, trajectory readout, GRU-D branch and cumulative-event readout."""
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config=copy.deepcopy(default_config() if config is None else config)
        c=self.config; data=c['data']
        if data['modalities']!=['EEG','ECG','EMG'] or data['num_classes']!=5:
            raise ValueError('This implementation expects EEG/ECG/EMG and five sleep stages')
        horizons=data['future_horizons']
        if not horizons or any(type(h) is not int or h<1 for h in horizons) or horizons!=sorted(set(horizons)):
            raise ValueError('future_horizons must be sorted, unique positive integer epoch offsets')
        if data['history_epochs']<2:
            raise ValueError('history_epochs must be at least 2')
        self.horizons=tuple(horizons)
        p,b=c['partial_observation'],c['belief_filter']
        self.state_model=build_mainline_model(c,model_class=RecursiveBeliefCarryCorrectWorldModel,
            extra_model_kwargs={
                'partial_filter_hidden_dim':int(p['hidden_dim']),
                'initial_freshness_decay':float(p['initial_freshness_decay']),
                'initial_uncertainty_age_scale':float(p['initial_uncertainty_age_scale']),
                'initial_uncertainty_horizon_scale':float(p['initial_uncertainty_horizon_scale']),
                'belief_hidden_dim':int(b['hidden_dim']), 'belief_max_delta':float(b['maximum_delta']),
                'belief_use_dynamics':bool(b.get('use_dynamics',True)),
                'belief_correction_mode':b.get('correction_mode','learned')})
        width=c['model']['d_model'];nf=len(c['physiology']['feature_names'])
        o=c['outcome_adapter'];d=c['direct'];f=c['fusion']
        self.outcome_adapter=TrajectoryOutcomeAdapter(width,3,5,nf,
            hidden_dim=o['hidden_dim'],maximum_stage_delta=o['maximum_stage_delta'],
            maximum_physiology_delta=o['maximum_physiology_delta'])
        self.direct_model=DirectObservationBranch('grud',3,self.horizons,nf,
            num_classes=5,feature_dim=d['feature_dim'],hidden_dim=d['hidden_dim'],dropout=d['dropout'])
        self.latent_adapter=LatentHazardSafeAdapter(width,3,5,nf,
            hidden_dim=f['hidden_dim'],maximum_physiology_residual=f['maximum_physiology_residual'],
            initial_direct_gate=f['initial_direct_gate'],dropout=f['dropout'])
        self.direct_event=DynamicEventHazardAdapter(d['hidden_dim'],3,5,nf,d['hidden_dim'],d['dropout'])
        self.event_correction=ReliabilityGatedEventCorrection(3,5)
        # The waveform decoder is inherited by the research architecture, but this
        # example trains only stage/feature/event outputs, so it has no loss here.
        self.state_model.waveform_decoder.requires_grad_(False)

    def forward(self, signals: torch.Tensor, present: torch.Tensor) -> dict:
        data=self.config['data']
        expected=(data['history_epochs'],3,data['sample_rate']*data['epoch_seconds'])
        if signals.ndim!=4 or tuple(signals.shape[1:])!=expected or present.shape!=signals.shape[:3]:
            raise ValueError(f'Expected signals [B,{expected[0]},3,{expected[2]}] and present [B,{expected[0]},3]')
        if present.dtype!=torch.bool:
            raise ValueError('present must be a boolean tensor')
        if present.device!=signals.device:
            raise ValueError('signals and present must be on the same device')
        if not torch.is_floating_point(signals) or not torch.isfinite(signals[present]).all():
            raise ValueError('Available signal samples must be finite floating-point values')
        signals=torch.where(present.unsqueeze(-1),signals,torch.zeros_like(signals))
        state=self.state_model.rollout_context_horizons(signals,present,self.horizons)
        adapted=self.outcome_adapter(state)
        direct=self.direct_model(signals,present)
        latent=self.latent_adapter(state,adapted['stage_logits'],adapted['future_physiology'],direct['future_physiology'])
        de=self.direct_event(direct,present,observation_age(present).to(signals.device),
                             torch.tensor(self.horizons,device=signals.device))
        corrected=self.event_correction(state,latent['stage_logits'],latent['interval_hazard'],de['interval_hazard'])
        return {'stage_logits':latent['stage_logits'], 'stage_probabilities':latent['stage_logits'].softmax(-1),
                'future_physiology':latent['future_physiology'],
                'transition_risk':corrected['transition_risk'],'interval_hazard':corrected['interval_hazard']}

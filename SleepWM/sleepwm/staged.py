"""Continue the paper's state/readout stages from a compatible reference model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile

import torch

from . import SleepWM, default_config
from .builders import build_mainline_model
from .data import load_statistics
from .runner import read_statistics, save_checkpoint as save_combined, write_json
from .staged_support import (classification_metrics, data_loader, load_checkpoint,
                             physio_feature_sequence_dataset, resolve_device)

STAGES = {
    'carry_align': ('partial_observation_filter', ()),
    'carry_final': ('partial_observation_filter', ('carry_align',)),
    'belief': ('recursive_belief_filter', ('carry_final',)),
    'outcome': ('trajectory_outcome_adapter', ('belief',)),
    'direct': ('direct_observation_branch', ()),
    'direct_event': ('direct_event_adapter', ('direct',)),
    'fusion': ('latent_hazard_safe_adapter', ('belief', 'outcome', 'direct')),
    'event': ('sleepwm_event_correction', ('belief', 'outcome', 'direct', 'direct_event', 'fusion')),
}
SEEDED = {'direct', 'direct_event', 'fusion', 'event'}
SCOPE = 'Staged state/readout training from a compatible reference model.'


def merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def checkpoint_path(root, stage, seed):
    folder = root / stage
    return (folder / f'seed_{seed}' if stage in SEEDED else folder) / 'best.pt'


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_config(config, args):
    config = copy.deepcopy(config)
    config.setdefault('data', {}).update(manifest_path=str(Path(args.manifest).resolve()),
        normalization_path=str(Path(args.statistics).resolve()))
    config['data'].setdefault('sequence_stride', 8)
    config.setdefault('experiment', {}).update(id='SleepWM_staged', seed=args.seed)
    train = config.setdefault('train', {})
    train.update(device=args.device, num_workers=0, max_batches=args.max_batches)
    if args.batch_size is not None:
        train['batch_size'] = args.batch_size
    train.setdefault('batch_size', 24)
    if args.epochs is not None:
        train['epochs'] = args.epochs
    return config


def check_normalization(checkpoint, statistics, reference_statistics=None):
    source = checkpoint.get('normalization')
    if source is None:
        if reference_statistics is None:
            raise ValueError('The reference needs normalization metadata or --reference-statistics from its training run')
        source = load_statistics(reference_statistics)
    elif reference_statistics is not None and source != load_statistics(reference_statistics):
        raise ValueError('Reference checkpoint and reference statistics do not match')
    if source != statistics:
        raise ValueError('Normalization mismatch: prepare data with the frozen reference statistics')


@torch.inference_mode()
def reference_score(reference, config, device):
    teacher = build_mainline_model(config).to(device).eval()
    teacher.load_state_dict(reference['model_state'], strict=True)
    dataset = physio_feature_sequence_dataset(config, 'val')
    logits, labels = [], []
    for batch in data_loader(dataset, config, shuffle=False):
        result = teacher.rollout_context(batch['history_signals'].to(device), batch['history_present'].to(device))
        logits.append(result['stage_logits'].cpu())
        labels.append(batch['future_labels'][:, :result['stage_logits'].shape[1]])
    return classification_metrics(torch.cat(logits), torch.cat(labels), 5)['macro_f1']


def verified_dependency(root, stage, seed, statistics, recorded, reference_data):
    path = checkpoint_path(root, stage, seed)
    entry = recorded['stages'].get(stage)
    if entry is None or entry['sha256'] != fingerprint(path):
        raise ValueError(f'{stage} checkpoint is missing from this run or has changed since selection')
    payload = load_checkpoint(path)
    check_normalization(payload, statistics)
    if payload['config']['experiment']['seed'] != seed:
        raise ValueError(f'{stage} checkpoint uses a different seed')
    for key in ('history_epochs', 'future_horizons', 'modalities', 'sample_rate'):
        if payload['config']['data'][key] != reference_data[key]:
            raise ValueError(f'{stage} checkpoint has a different {key}')
    return payload


def assemble(root, seed, statistics, statistics_sha, metadata, reference_data):
    names = ['belief', 'outcome', 'direct', 'fusion', 'direct_event', 'event']
    payloads = {name: verified_dependency(root, name, seed, statistics, metadata, reference_data)
                for name in names}
    config = merge(default_config(), payloads['belief']['config'])
    config['outcome_adapter'] = payloads['outcome']['config']['outcome_adapter']
    config['fusion'] = payloads['fusion']['config']['fusion']
    config['direct'] = payloads['direct']['config']['model']
    for payload in payloads.values():
        if payload['config']['data']['future_horizons'] != config['data']['future_horizons']:
            raise ValueError('Stage checkpoints have different forecast horizons')
    model = SleepWM(config)
    for stage, module, key in (
        ('belief', 'state_model', 'model_state'), ('outcome', 'outcome_adapter', 'adapter_state'),
        ('direct', 'direct_model', 'model_state'), ('fusion', 'latent_adapter', 'adapter_state'),
        ('direct_event', 'direct_event', 'adapter_state'), ('event', 'event_correction', 'adapter_state'),
    ):
        getattr(model, module).load_state_dict(payloads[stage][key], strict=True)
    metadata = dict(metadata, training_mode='staged', claim_scope=SCOPE,
                    selected_epochs={name: payload['epoch'] for name, payload in payloads.items()})
    save_combined(root / 'combined.pt', model, config, statistics, statistics_sha, metadata)
    return metadata


def run(args):
    if args.list:
        for name, (_, dependencies) in STAGES.items():
            print(f'{name}: requires {", ".join(dependencies) or "reference configuration"}')
        print('all: run every stage and assemble combined.pt; assemble: combine existing stage checkpoints')
        return
    for name in ('reference', 'manifest', 'statistics', 'output_dir'):
        if not getattr(args, name):
            raise ValueError(f'--{name.replace("_", "-")} is required for staged training')
    if args.stage is None:
        raise ValueError('--stage is required; use --list to see the stages')
    for name in ('epochs', 'batch_size', 'max_batches'):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    device = resolve_device(args.device)
    statistics = load_statistics(args.statistics)
    _, statistics_sha = read_statistics(args.statistics)
    reference = load_checkpoint(args.reference)
    if not {'config', 'model_state'}.issubset(reference):
        raise ValueError('Reference checkpoint must contain config and model_state for the fully observed reference model')
    check_normalization(reference, statistics, args.reference_statistics)
    base = runtime_config(reference['config'], args)
    data = base['data']
    if (data.get('modalities') != ['EEG', 'ECG', 'EMG'] or data.get('sample_rate') != 128
            or data.get('epoch_seconds', 30) != 30 or data.get('num_classes') != 5):
        raise ValueError('Reference must use 128 Hz EEG/ECG/EMG, 30-second epochs and five stages')
    horizons = data['future_horizons']
    if horizons[:3] != [1, 2, 4] or horizons != sorted(set(horizons)):
        raise ValueError('Staged protocols require ordered forecast horizons beginning with [1, 2, 4]')
    if base['recursive']['direct_horizons'] != [1, 2, 4]:
        raise ValueError('Staged protocols require direct forecast anchors [1, 2, 4]')
    if data['history_epochs'] != 20:
        raise ValueError('Staged observation protocols require 20 history epochs (10 minutes)')
    # Architecture is checked even when resuming downstream stages.
    model = build_mainline_model(base)
    model.load_state_dict(reference['model_state'], strict=True)
    del model
    settings = json.loads(Path(__file__).with_name('stage_settings.json').read_text(encoding='utf-8'))
    if args.settings:
        override = json.loads(Path(args.settings).read_text(encoding='utf-8'))
        if set(override) - set(STAGES):
            raise ValueError('Settings contain unknown stage names')
        merge(settings, override)
    root = Path(args.output_dir).resolve()
    requested = list(STAGES) if args.stage == 'all' else ([] if args.stage == 'assemble' else [args.stage])
    for stage in requested:
        if (root / stage).exists():
            raise FileExistsError(f'Refusing to overwrite stage directory: {root / stage}')
        for dependency in STAGES[stage][1]:
            if dependency not in requested and not checkpoint_path(root, dependency, args.seed).is_file():
                raise FileNotFoundError(f'Missing {dependency} checkpoint: {checkpoint_path(root, dependency, args.seed)}')
    if args.stage in ('all', 'assemble') and (root / 'combined.pt').exists():
        raise FileExistsError(f'Refusing to overwrite {root / "combined.pt"}')
    identity = {'reference_sha256': fingerprint(args.reference), 'manifest_sha256': fingerprint(args.manifest),
                'normalization_sha256': statistics_sha, 'seed': args.seed}
    record_path = root / 'run.json'
    if record_path.exists():
        recorded = json.loads(record_path.read_text(encoding='utf-8'))
        if recorded['inputs'] != identity:
            raise ValueError('Existing run uses different reference/data/statistics/seed; choose a new output directory')
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Output directory is not an existing staged run or an empty directory')
        root.mkdir(parents=True, exist_ok=True)
        recorded = {'training_mode': 'staged', 'claim_scope': SCOPE, 'inputs': identity, 'stages': {},
                    'smoke_test': args.max_batches is not None, 'test_split_accessed': False}
        write_json(record_path, recorded)
    recorded['smoke_test'] |= args.max_batches is not None
    if any(stage in ('carry_align', 'carry_final', 'belief') for stage in requested):
        score_config = runtime_config(merge(copy.deepcopy(base), {'train': {'batch_size': args.batch_size or 24}}), args)
        full_score = reference_score(reference, score_config, device)
    else:
        full_score = None
    with tempfile.TemporaryDirectory(prefix='sleepwm-staged-') as temporary:
        temporary = Path(temporary)
        for stage in requested:
            module_name, dependencies = STAGES[stage]
            config = runtime_config(merge(copy.deepcopy(base), settings[stage]), args)
            config['train']['output_dir'] = str(root / stage)
            config['baseline'] = {'checkpoint_path': str(Path(args.reference).resolve())}
            for section in ('partial_observation', 'belief_filter'):
                if section in config and full_score is not None:
                    explicit = settings[stage].get(section, {}).get('full_reference_macro_f1')
                    config[section]['full_reference_macro_f1'] = float(full_score if explicit is None else explicit)
            path = lambda name: str(checkpoint_path(root, name, args.seed))
            if stage == 'carry_align':
                config['partial_observation'].pop('initial_checkpoint', None)
            if stage == 'carry_final':
                config['partial_observation']['initial_checkpoint'] = path('carry_align')
            if stage == 'belief':
                config['belief_filter']['initial_checkpoint'] = path('carry_final')
            if stage == 'outcome':
                config['po3'] = {'checkpoint_path': path('belief')}
            if stage == 'fusion':
                config['checkpoints'] = {'po3': path('belief'), 'outcome': path('outcome'), 'direct': path('direct')}
            for dependency in dependencies:
                verified_dependency(root, dependency, args.seed, statistics, recorded, data)
            config_file = temporary / f'{stage}.json'
            write_json(config_file, config)
            if stage in ('direct_event', 'event'):
                inherited_stage = 'direct' if stage == 'direct_event' else 'fusion'
                inherited = load_checkpoint(path(inherited_stage))
                inherited['config'] = runtime_config(merge(copy.deepcopy(inherited['config']), settings[stage]), args)
                inherited['config']['train']['output_dir'] = str(root / stage)
                inherited_file = temporary / f'{stage}_input.pt'
                torch.save(inherited, inherited_file)
                argv = ['--seed', str(args.seed), '--device', args.device,
                        '--epochs', str(config['train']['epochs']), '--output-dir', str(root / stage), '--skip-test']
                if stage == 'direct_event':
                    argv += ['--baseline-checkpoint', str(inherited_file)]
                else:
                    argv += ['--po17-checkpoint', str(inherited_file), '--po3-checkpoint', path('belief'),
                             '--outcome-checkpoint', path('outcome'), '--direct-checkpoint', path('direct'),
                             '--direct-event-checkpoint', path('direct_event')]
            else:
                argv = ['--config', str(config_file), '--seed', str(args.seed)]
                if stage in ('direct', 'fusion'):
                    argv += ['--skip-test']
            print(f'\nStaged training: {stage}', flush=True)
            module = importlib.import_module(f'sleepwm.stages.{module_name}')
            old_argv = sys.argv
            try:
                sys.argv = [f'train.py --mode staged --stage {stage}', *argv]
                module.main()
            finally:
                sys.argv = old_argv
            output = checkpoint_path(root, stage, args.seed)
            selected = load_checkpoint(output)
            check_normalization(selected, statistics)
            metrics_path = output.parent / 'metrics.json'
            if stage == 'direct_event' and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
                metrics['baseline_checkpoint'] = path('direct')
                metrics_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
            recorded['stages'][stage] = {'checkpoint': str(output), 'sha256': fingerprint(output),
                'selected_epoch': selected['epoch'], 'training_config': selected['config']['train'],
                'reference_macro_f1': full_score if stage in ('carry_align', 'carry_final', 'belief') else None}
            write_json(record_path, recorded)
    if args.stage in ('all', 'assemble'):
        metadata = assemble(root, args.seed, statistics, statistics_sha, recorded, data)
        recorded['selected_epochs'] = metadata['selected_epochs']
        write_json(record_path, recorded)
        print(f'Combined model saved: {root / "combined.pt"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--stage', choices=('all', *STAGES, 'assemble'))
    parser.add_argument('--reference', help='Trusted reference checkpoint containing config and model_state')
    parser.add_argument('--reference-statistics', help='Frozen normalization from reference training')
    parser.add_argument('--manifest')
    parser.add_argument('--statistics')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, help='Override epoch count for every selected stage')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-batches', type=int, help='Smoke test only: limit each training/validation pass')
    parser.add_argument('--settings', help='JSON mapping of stage names to configuration overrides')
    return run(parser.parse_args(argv))

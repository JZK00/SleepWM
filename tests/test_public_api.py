from __future__ import annotations


def test_public_sleepwm_namespace() -> None:
    import sleepwm

    assert sleepwm.__version__ == "1.0.0"
    assert sleepwm.RecursiveBeliefCarryCorrectWorldModel is not None

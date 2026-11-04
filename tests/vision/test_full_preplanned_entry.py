"""Current executable paths cannot silently run the legacy controller."""
import pytest
from experiments.graspgenx_sim import nero7_run, nero7_full_observation_run, run as historical_run
from experiments.graspgenx_sim.full_preplanned_entry import main


def test_visual_entry_refuses_before_opening_devices_or_simulation(capsys):
    assert main([]) == 2
    assert '未启动历史控制器' in capsys.readouterr().err


def test_historical_visual_entry_cannot_start_legacy_controller(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('legacy policy must not start')
    monkeypatch.setattr(nero7_run, '_historical_main_not_a_current_entry', fail)
    assert nero7_run.main([]) == 2
    assert nero7_full_observation_run.main([]) == 2
    assert historical_run.main([]) == 2


def test_visual_entry_rejects_legacy_mode():
    with pytest.raises(SystemExit) as error:
        main(['--planning-mode', 'legacy'])
    assert error.value.code == 2


def test_physics_entry_rejects_legacy_before_initializing_plant(tmp_path, monkeypatch):
    from experiments.continuous_intercept_sim import run as physics
    monkeypatch.setattr(physics, 'make_environment', lambda *a, **k: pytest.fail('plant started'))
    with pytest.raises(SystemExit) as error:
        physics.main(['--planning-mode', 'legacy', '--output', str(tmp_path / 'out')])
    assert error.value.code == 2 and not (tmp_path / 'out').exists()


def test_direct_physics_runner_rejects_legacy_before_creating_output(tmp_path):
    from types import SimpleNamespace
    from experiments.continuous_intercept_sim.run import run_case
    with pytest.raises(ValueError, match='only full_preplanned'):
        run_case(SimpleNamespace(planning_mode='legacy'), 'continuous', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()

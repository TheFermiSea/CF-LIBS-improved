import pytest

from cflibs.atomic.structures import Transition
from cflibs.inversion.identify.line_detection import _select_comb_transitions


def _transition(stage: int, wavelength_nm: float, a_ki: float, e_k_ev: float) -> Transition:
    return Transition(
        element="Si",
        ionization_stage=stage,
        wavelength_nm=wavelength_nm,
        A_ki=a_ki,
        E_k_ev=e_k_ev,
        E_i_ev=0.0,
        g_k=3,
        g_i=1,
        relative_intensity=None,
    )


def test_comb_transition_selection_prefers_boltzmann_bright_si_i_line() -> None:
    transitions = [
        _transition(stage=2, wavelength_nm=251.4, a_ki=1.0e9, e_k_ev=12.0),
        _transition(stage=1, wavelength_nm=288.2, a_ki=1.0e7, e_k_ev=5.0),
    ]

    selected = _select_comb_transitions(transitions, max_lines=1)

    assert selected[0].ionization_stage == 1
    assert selected[0].wavelength_nm == pytest.approx(288.2)

import numpy as np
import pytest

from src.flightpulse.drift import ks_drift, psi

RNG = np.random.default_rng(42)


def test_psi_is_zero_for_identical_distributions():
    x = RNG.normal(500, 150, 5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-6)


def test_psi_near_zero_for_bootstrap_resample():
    x = RNG.normal(500, 150, 5000)
    resample = RNG.choice(x, size=len(x), replace=True)
    assert psi(x, resample) < 0.1


def test_psi_flags_significant_shift():
    x = RNG.normal(500, 150, 5000)
    assert psi(x, x * 1.5) > 0.25


def test_psi_grades_severity():
    x = RNG.normal(500, 150, 20000)
    mild = psi(x, x * 1.1)
    assert mild < psi(x, x * 1.6)


def test_psi_is_finite_with_disjoint_support():
    a = RNG.normal(0, 1, 2000)
    b = RNG.normal(50, 1, 2000)
    assert np.isfinite(psi(a, b))


def test_ks_does_not_flag_identical():
    x = RNG.normal(0, 1, 2000)
    assert ks_drift(x, x)["drift"] is False


def test_ks_flags_shifted():
    a = RNG.normal(0, 1, 2000)
    b = RNG.normal(1.5, 1, 2000)
    assert ks_drift(a, b)["drift"] is True

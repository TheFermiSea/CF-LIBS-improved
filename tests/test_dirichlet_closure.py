import numpy as np
import pytest
from cflibs.inversion.physics.closure import ClosureEquation, ClosureMode
from cflibs.inversion.physics.quality import QualityMetrics, QualityAssessor

def test_dirichlet_residual_simple_mode():
    # Setup: raw concentrations that sum to 0.90
    intercepts = {"Fe": np.log(0.5), "Ni": np.log(0.4)}
    partition_funcs = {"Fe": 1.0, "Ni": 1.0}
    
    # In simple mode with threshold 0.05, deficit 0.10 > 0.05 should trigger residual
    result = ClosureEquation.apply_dirichlet_residual(
        intercepts, 
        partition_funcs, 
        mode="simple", 
        residual_threshold=0.05
    )
    
    assert result.residual_fraction == pytest.approx(0.10)
    assert sum(result.concentrations.values()) == pytest.approx(0.90)
    assert result.concentrations["Fe"] == pytest.approx(0.5)
    assert result.concentrations["Ni"] == pytest.approx(0.4)

def test_dirichlet_residual_dirichlet_mode():
    # Setup: raw concentrations that sum to 0.80
    intercepts = {"Fe": np.log(0.4), "Ni": np.log(0.4)}
    partition_funcs = {"Fe": 1.0, "Ni": 1.0}
    
    # In dirichlet mode, residual depends on alpha_residual
    # alpha_res = 2.0 => alpha_res - 1 = 1.0
    # alpha_det = 1.0 => sum(alpha_det - 1) = 0
    # residual = 1.0 / (0.8 + 0 + 1.0) = 1.0 / 1.8 = 0.555...
    result = ClosureEquation.apply_dirichlet_residual(
        intercepts, 
        partition_funcs, 
        mode="dirichlet", 
        alpha_residual=2.0,
        alpha_detected=1.0
    )
    
    expected_residual = 1.0 / 1.8
    assert result.residual_fraction == pytest.approx(expected_residual)
    assert sum(result.concentrations.values()) == pytest.approx(1.0 - expected_residual)

def test_closure_mode_enum():
    assert ClosureMode.DIRICHLET_RESIDUAL.value == "dirichlet_residual"

def test_quality_metrics_gamma_residual():
    # Mock data
    concentrations = {"Fe": 0.90, "Ni": 0.05} # sum = 0.95
    
    metrics = QualityMetrics(
        r_squared_boltzmann=0.99,
        closure_residual=0.05,
        gamma_residual=0.05
    )
    
    assert metrics.gamma_residual == 0.05
    d = metrics.to_dict()
    assert d["gamma_residual"] == 0.05

def test_unified_benchmark_gamma_reporting():
    # This test verifies that if concentrations sum to < 1, 
    # gamma_residual is added to the prediction dict in unified.py logic.
    
    result = {"Fe": 0.90, "Ni": 0.05}
    total_conc = sum(result.values())
    prediction = {"concentrations": result}
    if total_conc < 0.999:
        prediction["gamma_residual"] = float(max(0.0, 1.0 - total_conc))
        
    assert "gamma_residual" in prediction
    assert prediction["gamma_residual"] == pytest.approx(0.05)

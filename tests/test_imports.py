import sys

import mp4eo_optimization


def test_core_api_imports_without_loading_solver_backend():
    assert hasattr(mp4eo_optimization, "MP4EOInstance")
    assert hasattr(mp4eo_optimization, "BLIPMapper")
    assert hasattr(mp4eo_optimization, "QuditsMapper")
    assert "mp4eo_optimization.mp4eo_solvers" not in sys.modules

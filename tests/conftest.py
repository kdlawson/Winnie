import numpy as np
import pytest

from winnie import backend # Winnie has to be imported before torch for it to be able to set PYTORCH_ENABLE_MPS_FALLBACK
import torch


def _available_devices():
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


@pytest.fixture(params=_available_devices())
def device(request):
    """Every device that is available on the system running the tests."""
    return request.param


@pytest.fixture
def rtol(device):
    """
    Tolerance (relative to the peak of the reference) for comparing results against float64 references: float32
    calculations (MPS) are much less precise than float64 ones (CPU, CUDA).
    """
    return 1e-9 if backend.resolve_dtype(device, np.float64) == torch.float64 else 5e-5


def assert_close(actual, desired, rtol, mask=None, scale=None):
    """
    Asserts that actual matches desired to within rtol times scale (defaulting to the peak of desired), ignoring
    non-finite values (which must be in the same locations).
    """
    actual, desired = np.asarray(actual, dtype=np.float64), np.asarray(desired, dtype=np.float64)
    assert actual.shape == desired.shape
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(desired))
    good = np.isfinite(desired) if mask is None else (np.isfinite(desired) & mask)
    peak = np.max(np.abs(desired[good])) if scale is None else scale
    np.testing.assert_allclose(actual[good], desired[good], rtol=0, atol=rtol*peak)

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from winnie import backend


def test_auto_device_priority(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert backend.auto_device().type == 'mps'
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert backend.auto_device().type == 'cuda'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert backend.auto_device().type == 'cpu'


def test_default_device_can_be_changed(monkeypatch):
    monkeypatch.setattr(backend, '_default_device', None)
    backend.set_default_device('cpu')
    assert backend.get_device() == torch.device('cpu')
    assert backend.get_device('cpu') == torch.device('cpu')
    backend.set_default_device(None) # back to automatic selection
    assert backend.get_device() == backend.auto_device()


def test_use_device_temporarily_changes_the_default(monkeypatch):
    monkeypatch.setattr(backend, '_default_device', None)
    original = backend.get_default_device()
    with backend.use_device('cpu'):
        assert backend.get_default_device() == torch.device('cpu')
        with backend.use_device(None): # leaves the default alone
            assert backend.get_default_device() == torch.device('cpu')
    assert backend.get_default_device() == original
    with pytest.raises(ValueError):
        with backend.use_device('cpu'):
            raise ValueError
    assert backend.get_default_device() == original # restored even if there was an error


def test_env_var_sets_default_device(monkeypatch):
    monkeypatch.setattr(backend, '_default_device', None)
    monkeypatch.setenv('WINNIE_DEVICE', 'cpu')
    assert backend.get_default_device() == torch.device('cpu')


def test_unavailable_device_is_reported():
    if torch.cuda.is_available():
        pytest.skip('CUDA is available')
    assert not backend.device_available('cuda')
    with pytest.raises(RuntimeError, match='not available'):
        backend.get_device('cuda')


def test_resolve_dtype():
    assert backend.resolve_dtype('cpu', np.float64) == torch.float64
    assert backend.resolve_dtype('cpu', np.float32) == torch.float32
    assert backend.resolve_dtype('cpu', np.float32, np.float64) == torch.float64
    assert backend.resolve_dtype('cpu', torch.float64) == torch.float64
    assert backend.resolve_dtype('cpu', np.dtype('>f8')) == torch.float64 # big-endian, as read from FITS files
    assert backend.resolve_dtype('cpu', np.dtype('>f4')) == torch.float32
    if torch.backends.mps.is_available():
        assert backend.resolve_dtype('mps', np.float64) == torch.float32 # MPS does not support float64


def test_to_tensor_handles_arrays_torch_cannot_wrap(device):
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    awkward = [base.astype('>f4'), # big-endian, as read from FITS files
               np.broadcast_to(base[0], (3, 4)), # read-only
               base[::-1], # negative strides
               base.T]
    for arr in awkward:
        np.testing.assert_array_equal(backend.to_numpy(backend.to_tensor(arr, device, torch.float32)), arr)


def test_to_tensor_casts_before_transfer(device):
    t = backend.to_tensor(np.ones(3, dtype=np.float64), device, backend.resolve_dtype(device, np.float64))
    assert t.device.type == torch.device(device).type
    assert t.dtype == backend.resolve_dtype(device, np.float64)


def test_chunk_ranges():
    assert backend.chunk_ranges(10, 4, 16) == [(0, 4), (4, 8), (8, 10)]
    assert backend.chunk_ranges(3, 100, 10) == [(0, 1), (1, 2), (2, 3)] # always at least one item per chunk
    assert backend.chunk_ranges(0, 4, 16) == []


def test_set_cpu_threads():
    n = torch.get_num_threads()
    try:
        backend.set_cpu_threads(-1) # non-positive values leave torch's setting alone
        assert torch.get_num_threads() == n
        backend.set_cpu_threads(2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(n)


def _run_python(code, env_updates):
    env = {k: v for k, v in os.environ.items() if k != 'PYTORCH_ENABLE_MPS_FALLBACK'}
    env.update(env_updates)
    return subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True, env=env)


def test_mps_fallback_is_enabled_before_torch_is_imported():
    code = "import os, sys; import winnie; print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'], 'torch' in sys.modules)"
    result = _run_python(code, {})
    assert result.stdout.split()[-2:] == ['1', 'True'], result.stderr[-500:]


def test_user_setting_of_mps_fallback_is_respected():
    code = "import os; import winnie; print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])"
    result = _run_python(code, {'PYTORCH_ENABLE_MPS_FALLBACK': '0'})
    assert result.stdout.split()[-1] == '0', result.stderr[-500:]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is not available')
def test_warns_if_torch_was_imported_first():
    code = ("import warnings, torch\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import winnie\n"
            "print(any('PYTORCH_ENABLE_MPS_FALLBACK' in str(x.message) for x in w))")
    result = _run_python(code, {})
    assert result.stdout.split()[-1] == 'True', result.stderr[-500:]

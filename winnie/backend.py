"""
Device and dtype handling for Winnie's PyTorch-based calculations.

The default device is the best one available, with priority 'mps' > 'cuda' > 'cpu'. It can be changed
 - for the whole process with set_default_device() (or by setting the WINNIE_DEVICE environment variable before
   importing Winnie),
 - for a single SpaceRDI / SpaceConvolution instance with its ``device`` argument, or
 - for a single call with the ``device`` argument accepted by most functions in Winnie.

Any ``device`` argument may be a string (e.g., 'cpu', 'mps', 'cuda:1'), a torch.device, or None to use the default.

Note: when the default device is 'mps', unsupported operations fall back to the CPU because winnie/__init__.py sets
PYTORCH_ENABLE_MPS_FALLBACK=1 prior to importing torch. To opt out, set PYTORCH_ENABLE_MPS_FALLBACK=0 before
starting Python.
"""
import os
import contextlib
import numpy as np
import torch

_DEVICE_ENV_VAR = 'WINNIE_DEVICE'
_default_device = None

_NP_DTYPES = {torch.float32: np.float32, torch.float64: np.float64}


def _is_available(device):
    if device.type == 'mps':
        return torch.backends.mps.is_available()
    if device.type == 'cuda':
        return torch.cuda.is_available()
    return True


def device_available(device):
    """
    Returns whether device (a string or torch.device) is available on this system.
    """
    try:
        return _is_available(torch.device(device))
    except RuntimeError:
        return False


def auto_device():
    """
    Returns the best available device, with priority 'mps' > 'cuda' > 'cpu'.
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_device(device=None):
    """
    Resolves device (a string, torch.device, or None) to an available torch.device. If None, the current default
    device is returned.
    """
    if device is None:
        return get_default_device()
    device = torch.device(device)
    if not _is_available(device):
        raise RuntimeError(f"The requested device '{device}' is not available on this system. Use device=None "
                           "(or set_default_device(None)) to select the best available device automatically.")
    return device


def get_default_device():
    """
    Returns the default device used when ``device`` is None. This is the device named by the WINNIE_DEVICE
    environment variable if set, otherwise the best available device ('mps' > 'cuda' > 'cpu').
    """
    global _default_device
    if _default_device is None:
        env_device = os.environ.get(_DEVICE_ENV_VAR)
        _default_device = get_device(env_device) if env_device else auto_device()
    return _default_device


def set_default_device(device=None):
    """
    Sets the default device used by Winnie when ``device`` is None. Passing None restores automatic selection.
    Instances that were created with an explicit ``device`` are unaffected.
    """
    global _default_device
    _default_device = None if device is None else get_device(device)
    return get_default_device()


def resolve_dtype(device, *dtypes):
    """
    Returns the floating point dtype (torch.float32 or torch.float64) to use for calculations on device given the
    dtypes of the inputs: float64 is retained if any input is float64 and the device supports it. MPS does not
    support float64, so calculations there are always carried out in float32.
    """
    if get_device(device).type == 'mps':
        return torch.float32
    for dtype in dtypes:
        if dtype is None:
            continue
        # issubdtype (unlike ==) ignores byte order, and float64 arrays read from FITS files are big-endian
        is_64 = (dtype == torch.float64) if isinstance(dtype, torch.dtype) else np.issubdtype(np.dtype(dtype), np.float64)
        if is_64:
            return torch.float64
    return torch.float32


def to_tensor(x, device=None, dtype=None):
    """
    Converts an array-like (or tensor) to a tensor on device with the optional torch dtype. Casting to dtype is done
    before any transfer, since MPS cannot hold float64 data. Arrays that torch cannot wrap directly (e.g., big-endian
    arrays as read from FITS files, read-only arrays, or arrays with negative strides) are copied first.
    """
    device = get_device(device)
    if isinstance(x, torch.Tensor):
        t = x
    else:
        a = np.asarray(x)
        if not a.dtype.isnative:
            a = a.astype(a.dtype.newbyteorder('='))
        if not a.flags.writeable or any(s < 0 for s in a.strides):
            a = a.copy()
        t = torch.from_numpy(a)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t.to(device)


def to_numpy(t, dtype=None):
    """
    Converts a tensor to a numpy array (on the CPU), optionally casting to a numpy dtype.
    """
    a = t.detach().cpu().numpy()
    return a if dtype is None else a.astype(dtype, copy=False)


@contextlib.contextmanager
def use_device(device):
    """
    Context manager that temporarily makes device the default device. This lets code that is called without a ``device``
    argument (e.g., user-supplied callables) use the device of the object that calls it. If device is None, the default
    device is left unchanged.
    """
    global _default_device
    previous = _default_device
    if device is not None:
        _default_device = get_device(device)
    try:
        yield
    finally:
        _default_device = previous


def set_cpu_threads(ncores):
    """
    Sets the number of CPU threads that torch uses for calculations on the CPU (a process-wide setting; see
    torch.set_num_threads). Values of ncores that are not positive integers (e.g., the default of -1 for SpaceRDI) leave
    torch's default thread settings unchanged.
    """
    if ncores is not None and ncores > 0:
        torch.set_num_threads(int(ncores))


def clear_device_cache(device=None):
    """
    Releases cached memory held by torch's allocator for device back to the system. This is not needed for normal use;
    it is only useful for freeing memory for other programs after large calculations.
    """
    device = get_device(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def chunk_ranges(n, bytes_per_item, max_bytes=2**30):
    """
    Splits range(n) into consecutive (start, stop) ranges such that each holds at most max_bytes at
    bytes_per_item per item (but always at least one item). Used to bound the peak memory of batched calculations.
    """
    step = int(max(1, min(n, max_bytes // max(1, int(bytes_per_item)))))
    return [(i, min(i + step, n)) for i in range(0, n, step)]

from types import SimpleNamespace

from winnie import RCRDI, SpaceRDI
from winnie.roman import RCConvolution, RCReduction


def test_hook_classes():
    # JWST keeps the getattr defaults (SpaceReduction/SpaceConvolution); only RC overrides them.
    assert not hasattr(SpaceRDI, 'reduction_class') and not hasattr(SpaceRDI, 'convolution_class')
    assert RCRDI.reduction_class is RCReduction and RCRDI.convolution_class is RCConvolution


def test_load_convolver_forwarding():
    calls = []
    obj = SimpleNamespace(concat='X', convolver_args={'a': 1},
                          convolver=SimpleNamespace(load_concat=lambda *args, **kw: calls.append((args, kw))))
    SpaceRDI._load_convolver(obj, cropped_shape=(5, 6), coron_offsets='off')
    SpaceRDI._load_convolver(obj)
    assert calls == [(('X',), dict(cropped_shape=(5, 6), coron_offsets='off', a=1)), (('X',), dict(a=1))]

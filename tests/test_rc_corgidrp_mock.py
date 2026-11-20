import os
import warnings

import numpy as np
import pytest

from winnie import RCRDI
from winnie.roman import RCDatabase

mocks = pytest.importorskip('corgidrp.mocks')

# corgidrp's own L3 writer (create_psfsub_dataset): Gaussian star at the array center, planet at PL_SEP px due north
# on sky, STARLOCX/Y, PA_APER and NORTHANG set per frame; every frame has the same VISITID.
SHAPE = (61, 61)
PL_SEP = 10.0
SCI_PAS = (0., 30.)
CONTRAST = 1e-2


@pytest.fixture(scope='module')
def mock_db(tmp_path_factory):
    d = str(tmp_path_factory.mktemp('corgidrp_mock'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sci, ref = mocks.create_psfsub_dataset(len(SCI_PAS), 1, [*SCI_PAS, 0.], data_shape=list(SHAPE),
                                               noise_amp=0.01, pl_contrast=CONTRAST, pl_sep=PL_SEP)
    files = []
    for i, frame in enumerate([*sci, *ref]):
        # The mock sets NAXIS1/2 on a data-less primary header, which astropy refuses to write.
        for k in ('NAXIS1', 'NAXIS2'):
            frame.pri_hdr.remove(k, ignore_missing=True)
        name = f'cgi_0200001001001001001_2027010{i}t0000000_l3_.fits'
        frame.save(filedir=d, filename=name)
        files.append(os.path.join(d, name))
    db = RCDatabase(os.path.join(d, 'out'))
    db.add_concat(files[:len(SCI_PAS)], files[len(SCI_PAS):])
    return db


def test_mock_headers(mock_db):
    tab = mock_db.obs['RC_0']
    center = (np.array(SHAPE) - 1) / 2
    assert list(tab['TYPE']) == ['SCI', 'SCI', 'REF']
    assert np.allclose(tab['STARX'], center[1]) and np.allclose(tab['STARY'], center[0])
    assert np.allclose(tab['PIXSCALE'], 0.0218) and np.allclose(tab['CWAVEL'], 0.575)


def test_mock_rolls_not_combined(mock_db):
    # Same VISITID on every frame: the two SCI rolls and the REF must stay separate images.
    rdi = RCRDI(mock_db, pad_data=None, verbose=False)
    rdi.load_concat(0)
    assert rdi.imcube_sci.shape == (2, *SHAPE) and rdi.imcube_ref.shape == (1, *SHAPE)
    assert np.allclose(rdi._posangs_sci, SCI_PAS) and np.allclose(rdi._posangs_ref, 0.)
    assert len(set(rdi._visit_ids_sci)) == 2
    assert rdi.data_ext == 'l3'


def test_mock_planet_north(mock_db):
    # The mock places the planet north on sky at every roll, so derotated RDI must put it straight up.
    rdi = RCRDI(mock_db, pad_data=None, verbose=False)
    rdi.load_concat(0)
    out = rdi.run_rdi()
    cs = out.c_star
    y, x = np.indices(out.im.shape)
    m = np.hypot(x - cs[0], y - cs[1]) < 3
    py, px = np.unravel_index(np.nanargmax(np.where(m, -np.inf, out.im)), out.im.shape)
    assert np.allclose([px - cs[0], py - cs[1]], [0., PL_SEP], atol=1)
    # Star removed well below the planet; planet keeps most of its flux (100*CONTRAST peak in the mock).
    assert np.nanmax(np.abs(out.im[m])) < 0.1*out.im[py, px]
    assert 0.8 < out.im[py, px]/(100*CONTRAST) < 1.2
    for im, pa in zip(out.rolls, SCI_PAS):
        assert np.isclose(im[py, px], out.im[py, px], rtol=0.2)

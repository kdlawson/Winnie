import os

import numpy as np
import pytest

import conftest as c
import winnie.roman as roman
from winnie.roman import RCDatabase, find_rc_star


@pytest.fixture
def no_starloc_files(tmp_path):
    """One SCI visit without STARLOCX/Y, plus a sat-spot frame of the same visit."""
    rng = np.random.default_rng(1)
    sci = str(tmp_path / 'sci.fits')
    spot = str(tmp_path / 'spot.fits')
    c._write_rc_file(sci, 0.0, '0000100100100100100', rng, starloc=False)
    c._write_rc_file(spot, 0.0, '0000100100100100100', rng, starloc=False)
    from astropy.io import fits
    with fits.open(spot, mode='update') as hdul:
        hdul[1].header['SATSPOTS'] = 1
    return sci, spot


def test_add_concat_runs_find_star(no_starloc_files, tmp_path, monkeypatch):
    sci, spot = no_starloc_files
    calls = []

    def fake_find(files):
        calls.append(sorted(files))
        return {sci: np.array([30.4, 29.6])}

    monkeypatch.setattr(roman, 'find_rc_star', fake_find)
    db = RCDatabase(str(tmp_path / 'out'))
    tab = db.obs[db.add_concat([sci, spot], [])]
    # Sat-spot frames go to find_star but not into the table.
    assert calls == [sorted([sci, spot])]
    assert list(tab['FITSFILE']) == [sci]
    assert (tab['STARX'][0], tab['STARY'][0]) == (30.4, 29.6)


def test_add_concat_needs_star_info(no_starloc_files, tmp_path):
    sci, _ = no_starloc_files
    db = RCDatabase(str(tmp_path / 'out'))
    with pytest.raises(ValueError, match='satellite spot'):
        db.add_concat([sci], [])


def test_find_rc_star_corgidrp_mock(tmp_path):
    # Real corgidrp find_star on corgidrp's own mock sat-spot sequence (default L3 headers).
    mocks = pytest.importorskip('corgidrp.mocks')
    true_xy = (30.2, 29.7)
    ds = mocks.create_satellite_spot_observing_sequence(2, 3, image_shape=(61, 61), star_center=true_xy)
    files = []
    for i, frame in enumerate(ds):
        # The mock sets NAXIS1/2 on a data-less primary header, which astropy refuses to write.
        for k in ('NAXIS1', 'NAXIS2'):
            frame.pri_hdr.remove(k, ignore_missing=True)
        frame.save(filedir=str(tmp_path), filename=f'frame{i}_l3_.fits')
        files.append(os.path.join(str(tmp_path), f'frame{i}_l3_.fits'))
    c_stars = find_rc_star(files)
    assert len(c_stars) == 2
    for xy in c_stars.values():
        assert np.allclose(xy, true_xy, atol=0.1)

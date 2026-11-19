import os

import numpy as np
import pytest
from astropy import wcs
from astropy.io import fits

import conftest as c
from winnie import RCRDI
from winnie.roman import RCDatabase, RCReduction


def _centroid(im, xy, hw=2):
    x0, y0 = np.round(xy).astype(int)
    box = np.nan_to_num(im[y0-hw:y0+hw+1, x0-hw:x0+hw+1])
    y, x = np.mgrid[y0-hw:y0+hw+1, x0-hw:x0+hw+1]
    return np.array([np.sum(x*box), np.sum(y*box)]) / np.sum(box)


def _north_up_planet(c_star):
    # North up, east left: PA is measured from +y toward -x.
    pa = np.deg2rad(c.PLANET_PA_DEG)
    return c_star[0] - c.PLANET_SEP_PX*np.sin(pa), c_star[1] + c.PLANET_SEP_PX*np.cos(pa)


@pytest.fixture
def rdi(rc_fixture, tmp_path):
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat(rc_fixture['sci'], rc_fixture['ref'])
    rdi = RCRDI(db, pad_data='auto', verbose=False)
    rdi.load_concat(0)
    return rdi


def test_save_and_register(rdi):
    products = rdi.run_rdi(save_products=True)
    assert isinstance(products, RCReduction)
    assert products.filename == f'{rdi.output_dir}RC_0_rdi_psfsub_i2d.fits'
    assert os.path.isfile(products.filename)
    assert list(rdi.database.red['RC_0']['FITSFILE']) == [products.filename]
    assert list(rdi.database.red['RC_0']['OUTPUT_EXT']) == ['rdi_psfsub']
    with pytest.raises(OSError):
        rdi.run_rdi(save_products=True)
    rdi.overwrite = True
    rdi.run_rdi(save_products=True)
    assert len(rdi.database.red['RC_0']) == 1


def test_star_suppression(rdi):
    res = rdi.run_rdi(derotate=False, collapse_rolls=False, return_res_only=True)[0][:, 0]
    y, x = np.indices(rdi.imcube_sci.shape[-2:])
    core = np.hypot(x - rdi.c_star[0], y - rdi.c_star[1]) < 6
    for sci, r in zip(rdi.imcube_sci, res):
        assert np.nansum(np.abs(r[core])) < 0.1*np.nansum(np.abs(sci[core]))


def test_planet_derotated(rdi):
    products = rdi.run_rdi()
    assert products.derotated and products.primary_header['DEROT']
    xy = _north_up_planet(products.c_star)
    for im in [products.im, *products.rolls]:
        assert np.allclose(_centroid(im, xy), xy, atol=0.2)


def test_planet_not_derotated(rdi):
    products = rdi.run_rdi(derotate=False)
    assert not products.derotated
    for im, roll in zip(products.rolls, c.SCI_ROLLS):
        xy = c.planet_xy(roll, products.c_star)
        assert np.allclose(_centroid(im, xy), xy, atol=0.2)


def test_derotated_wcs(rdi):
    products = rdi.run_rdi()
    xy = _north_up_planet(products.c_star)
    for h in [products.combined_image_header, *products.image_headers]:
        w = wcs.WCS(h, naxis=[1, 2])
        ra0, dec0 = w.all_pix2world(*products.c_star, 0)
        ra, dec = w.all_pix2world(*xy, 0)
        dra, ddec = (ra - ra0)*3600, (dec - dec0)*3600
        assert np.isclose(np.rad2deg(np.arctan2(dra, ddec)) % 360, c.PLANET_PA_DEG, atol=1)
        assert np.isclose(np.hypot(dra, ddec), c.PLANET_SEP_PX*c.PXSCALE, rtol=0.01)


def test_load_saved(rdi):
    products = rdi.run_rdi(save_products=True)
    loaded = RCReduction(file_to_load=products.filename)
    assert loaded.output_ext == 'rdi_psfsub'
    assert loaded.derotated
    assert np.allclose(loaded.im, products.im, equal_nan=True)
    assert np.allclose(loaded.rolls, products.rolls, equal_nan=True)
    assert np.allclose(loaded.c_star, products.c_star)
    assert np.isclose(loaded.pxscale, products.pxscale)


def test_save_instance(rdi):
    rdi.save_instance = True
    products = rdi.run_rdi(save_products=True)
    assert 'WINNIE_PICKLED' in fits.open(products.filename)
    restored = RCRDI(from_fits=products.filename)
    assert np.allclose(restored.imcube_sci, rdi.imcube_sci, equal_nan=True)
    assert np.allclose(restored.imcube_ref, rdi.imcube_ref, equal_nan=True)
    assert np.allclose(restored.c_star, rdi.c_star)
    assert rdi.efficient_saving

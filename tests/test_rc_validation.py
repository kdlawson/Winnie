import numpy as np
import pytest

import conftest as c
from winnie import RCRDI
from winnie.roman import RCDatabase

SIGMA = 2.5 / np.sqrt(8*np.log(2))  # fixture planet Gaussian
AP_RAD = 3


def _aperture_flux(im, xy, rad=AP_RAD):
    y, x = np.indices(im.shape)
    return np.nansum(im[np.hypot(x - xy[0], y - xy[1]) < rad])


def _planet_truth_flux(rad=AP_RAD):
    return c.PLANET_CONTRAST * 2*np.pi*SIGMA**2 * (1 - np.exp(-rad**2 / (2*SIGMA**2)))


def _north_up_planet(c_star):
    pa = np.deg2rad(c.PLANET_PA_DEG)
    return c_star[0] - c.PLANET_SEP_PX*np.sin(pa), c_star[1] + c.PLANET_SEP_PX*np.cos(pa)


@pytest.fixture
def rdi(rc_fixture, tmp_path):
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat(rc_fixture['sci'], rc_fixture['ref'])
    rdi = RCRDI(db, pad_data='auto', verbose=False)
    rdi.load_concat(0)
    return rdi


def test_planet_flux(rdi):
    products = rdi.run_rdi()
    xy = _north_up_planet(products.c_star)
    for im in [products.im, *products.rolls]:
        assert np.isclose(_aperture_flux(im, xy), _planet_truth_flux(), rtol=0.05)


def test_planet_flux_masked_optzone(rdi):
    optzones = rdi._optzones.copy()
    y, x = np.indices(optzones.shape[-2:])
    for roll in c.SCI_ROLLS:
        px, py = c.planet_xy(roll, rdi._c_star)
        optzones[..., np.hypot(x - px, y - py) < 4] = False
    rdi.set_zones(optzones, rdi._subzones)
    assert rdi.optzones.sum() < rdi._subzones.sum()
    products = rdi.run_rdi()
    xy = _north_up_planet(products.c_star)
    assert np.isclose(_aperture_flux(products.im, xy), _planet_truth_flux(), rtol=0.05)


def test_err_propagated(rdi):
    products = rdi.run_rdi()
    assert products.err.shape == products.im.shape
    assert products.err_rolls.shape == products.rolls.shape
    good = np.isfinite(products.im)
    assert np.all(np.isfinite(products.err[good]))
    assert np.all(products.err[good] > 0)


def test_hpfrdi(rdi):
    rdi.hpfrdi_presets()
    assert rdi.output_ext == 'hpfrdi_psfsub'
    res = rdi.run_rdi(derotate=False, return_res_only=True)[0][:, 0]
    y, x = np.indices(rdi.imcube_sci.shape[-2:])
    core = np.hypot(x - rdi.c_star[0], y - rdi.c_star[1]) < 6
    for sci, r in zip(rdi.imcube_sci, res):
        assert np.nansum(np.abs(r[core])) < 0.1*np.nansum(np.abs(sci[core]))
    products = rdi.run_rdi()
    assert np.isclose(_aperture_flux(products.im, _north_up_planet(products.c_star)), _planet_truth_flux(), rtol=0.05)


def test_jackknife(rdi):
    inds, rescube, errcube = rdi.jackknife_references(show_progress=False)
    nvis = len(np.unique(rdi._visit_ids_ref))
    assert len(inds) == rdi.imcube_ref.shape[0]
    assert rescube.shape == (nvis, *rdi.imcube_sci.shape)
    assert errcube.shape == rescube.shape
    assert np.all(np.isfinite(rescube[:, :, int(rdi.c_star[1]), int(rdi.c_star[0])]))

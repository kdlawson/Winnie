import pickle

import numpy as np
import pytest

import conftest as c
from winnie import RCRDI
from winnie.roman import RCConvolution, RCDatabase, RCInstrument, gaussian_psf_grid

FWHM = 2.5  # fixture planet Gaussian, px
OSAMP = 2
FOV = 21
AP_RAD = 3


def _centroid(im, xy, rad=AP_RAD):
    y, x = np.indices(im.shape)
    m = np.hypot(x - xy[0], y - xy[1]) < rad
    return np.array([np.sum(x[m]*im[m]), np.sum(y[m]*im[m])]) / np.sum(im[m])


def _aperture_flux(im, xy, rad=AP_RAD):
    y, x = np.indices(im.shape)
    return np.nansum(im[np.hypot(x - xy[0], y - xy[1]) < rad])


def _north_up_point_source(sep_px, pa_deg, flux=1.0, n=61, sigma=0.5):
    """
    North-up model at the detector pixel scale, star at the center: a narrow
    Gaussian of total flux `flux` (a single-pixel delta would be erased, since
    Winnie treats exact zeros as missing data when rotating).
    """
    c0 = (n - 1) / 2
    x0, y0 = c0 - sep_px*np.sin(np.deg2rad(pa_deg)), c0 + sep_px*np.cos(np.deg2rad(pa_deg))
    y, x = np.indices((n, n))
    model = np.exp(-((x - x0)**2 + (y - y0)**2) / (2*sigma**2))
    return model * flux / model.sum(), np.array([c0, c0])


@pytest.fixture
def rdi(rc_fixture, tmp_path):
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat(rc_fixture['sci'], rc_fixture['ref'])
    rdi = RCRDI(db, pad_data='auto', verbose=False)
    rdi.load_concat(0)
    return rdi


def _prepare(rdi, **kwargs):
    rdi.prepare_convolution(osamp=OSAMP, fov_pixels=FOV, grid_kwargs=dict(fwhm=FWHM), **kwargs)


def test_gaussian_psf_grid():
    inst = RCInstrument(c.PXSCALE, 0.575)
    psfs, psfs_nodist, polar, xy = gaussian_psf_grid(inst, osamp=OSAMP, fov_pixels=FOV, rvals_lamd=(0, 3, 6), ntheta=4)
    assert psfs.shape == (9, FOV*OSAMP + 1, FOV*OSAMP + 1)
    assert psfs is psfs_nodist
    assert np.allclose(psfs.sum(axis=(1, 2)), 1)
    assert np.allclose(np.hypot(*xy), polar[0])
    # theta is CCW from +Y: 90 deg points to -x.
    i = np.where((polar[1] == 90) & (polar[0] > 0))[0][0]
    assert xy[0, i] < 0 and np.isclose(xy[1, i], 0, atol=1e-12)


def test_prepare_convolution(rdi):
    _prepare(rdi)
    conv = rdi.convolver
    assert isinstance(conv, RCConvolution)
    assert np.array_equal(conv.posangs_sci, rdi._posangs_sci)
    assert np.allclose(conv.c_star, rdi.c_star)
    nsci = len(rdi.imcube_sci)
    assert conv.psf_inds_osamp.shape == (nsci, rdi.ny*OSAMP, rdi.nx*OSAMP)
    assert np.all(conv.coron_tmaps_osamp == 1)
    assert isinstance(conv.inst_stpsf, RCInstrument)


def test_psf_grid_cache(rdi):
    _prepare(rdi)
    cached = rdi.convolver.psfs.copy()
    rdi2 = RCRDI(rdi.database, pad_data='auto', verbose=False)
    rdi2.load_concat(0)
    calls = []
    rdi2.prepare_convolution(osamp=OSAMP, fov_pixels=FOV, grid_fn=lambda *a, **k: calls.append(1))
    assert not calls
    assert np.array_equal(rdi2.convolver.psfs, cached)


@pytest.mark.parametrize('sep_px, pa_deg', [(12.0, c.PLANET_PA_DEG), (6.0, 200.0)])
def test_convolve_point_source(rdi, sep_px, pa_deg):
    _prepare(rdi)
    model, c_model = _north_up_point_source(sep_px, pa_deg)
    cube = rdi.convolver.convolve_model(model, pxscale_in=rdi.pxscale, c_star_in=c_model)
    assert cube.shape == rdi.imcube_sci.shape
    for im, roll in zip(cube, rdi._posangs_sci):
        theta = np.deg2rad(pa_deg - roll)
        xy = rdi.c_star + sep_px*np.array([-np.sin(theta), np.cos(theta)])
        assert np.isclose(im.sum(), 1, rtol=0.01)
        assert np.hypot(*(_centroid(im, xy) - xy)) < 0.05


def test_forward_model(rdi):
    # Grid width chosen so grid PSF convolved with the model Gaussian (sigma 0.5) has the planet's FWHM.
    rdi.prepare_convolution(osamp=OSAMP, fov_pixels=FOV, grid_kwargs=dict(fwhm=np.sqrt(FWHM**2 - 8*np.log(2)*0.5**2)))
    data = rdi.run_rdi()
    model, c_model = _north_up_point_source(c.PLANET_SEP_PX, c.PLANET_PA_DEG, flux=c.PLANET_CONTRAST*2*np.pi*(FWHM/np.sqrt(8*np.log(2)))**2)
    rdi.set_circumstellar_model(raw_model=model, raw_model_pxscale=rdi.pxscale, raw_model_center=c_model)
    fm = rdi.run_rdi(forward_model=True)
    assert fm.output_ext.endswith('_fwdmod')
    pa = np.deg2rad(c.PLANET_PA_DEG)
    xy = data.c_star + c.PLANET_SEP_PX*np.array([-np.sin(pa), np.cos(pa)])
    assert np.isclose(_aperture_flux(fm.im, xy), _aperture_flux(data.im, xy), rtol=0.05)


def test_set_custom_grid(rdi):
    _prepare(rdi)
    model, c_model = _north_up_point_source(c.PLANET_SEP_PX, c.PLANET_PA_DEG)
    ref = rdi.convolver.convolve_model(model, pxscale_in=rdi.pxscale, c_star_in=c_model)
    conv = rdi.convolver
    psfs, inds, tmaps = conv.psfs.copy(), conv._psf_inds_osamp.copy(), conv._coron_tmaps_osamp.copy()
    rdi.convolver = None
    rdi.prepare_convolution(osamp=OSAMP, fov_pixels=FOV, prefetch_psf_grid=False)
    assert not rdi.convolver._grid_fetched
    rdi.convolver.set_custom_grid(psfs, inds, tmaps)
    assert np.allclose(rdi.convolver.convolve_model(model, pxscale_in=rdi.pxscale, c_star_in=c_model), ref)


def test_convolver_crop_and_pickle(rdi):
    _prepare(rdi)
    rdi.set_crop([41, 41])
    model, c_model = _north_up_point_source(c.PLANET_SEP_PX, c.PLANET_PA_DEG)
    rdi.set_circumstellar_model(raw_model=model, raw_model_pxscale=rdi.pxscale, raw_model_center=c_model)
    assert rdi.imcube_css.shape == rdi.imcube_sci.shape == (len(rdi._posangs_sci), 41, 41)
    conv = pickle.loads(pickle.dumps(rdi.convolver))
    assert np.array_equal(conv.psfs, rdi.convolver.psfs)
    assert conv.psf_inds_osamp.shape[-2:] == (41*OSAMP, 41*OSAMP)

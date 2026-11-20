import numpy as np
import pytest
from scipy import ndimage

import conftest as c
from winnie import RCRDI
from winnie.roman import RCDatabase

FWHM = 2.5  # fixture PSF (px)
OSAMP = 2
FOV = 21
SUB = 4  # raw model and data ring sampled at pxscale/SUB
RING = dict(r0=15.0, width=1.2, incl=50.0, pa=120.0, amp=4e-3)  # px, px, deg, deg (east of north), per px
PLANET_MASK = 6  # px, the fixture planet is not in the model


def ring_sb(xs, ys, r0, width, incl, pa, amp):
    """Inclined ring at north-up offsets (xs, ys) in px from the star (+y north, +x west); major axis along PA."""
    t = np.deg2rad(pa)
    along = -xs*np.sin(t) + ys*np.cos(t)
    across = (xs*np.cos(t) + ys*np.sin(t)) / np.cos(np.deg2rad(incl))
    return amp*np.exp(-(np.hypot(along, across) - r0)**2 / (2*width**2))


def detector_ring(roll, star_xy, shape=(c.NY, c.NX)):
    """The ring as the detector sees it at a roll: sampled at 1/SUB px, convolved with the PSF, binned."""
    ny, nx = shape
    yy, xx = (np.indices((ny*SUB, nx*SUB)) + 0.5) / SUB - 0.5
    dx, dy = xx - star_xy[0], yy - star_xy[1]
    # Detector offsets to north-up: PA on sky = detector angle + roll.
    r, th = np.hypot(dx, dy), np.arctan2(-dx, dy) + np.deg2rad(roll)
    sb = ndimage.gaussian_filter(ring_sb(-r*np.sin(th), r*np.cos(th), **RING), FWHM/np.sqrt(8*np.log(2))*SUB,
                                 mode='constant')
    return sb.reshape(ny, SUB, nx, SUB).mean(axis=(1, 3))


def raw_ring_model(n=240):
    """
    North-up raw model at pxscale/SUB, in flux per detector pixel (Winnie's raw-model units). n*SUB/OSAMP
    must be an integer: frebin rounds the resampled size, which otherwise stretches the model slightly.
    """
    c0 = (n - 1) / 2
    y, x = (np.indices((n, n)) - c0) / SUB
    return ring_sb(x, y, **RING), np.array([c0, c0])


@pytest.fixture(scope='module')
def ring_data(tmp_path_factory):
    d = tmp_path_factory.mktemp('rc_ring')
    rng = np.random.default_rng(1)
    sci, ref = [], []
    for i, roll in enumerate(c.SCI_ROLLS):
        p = str(d / f'rc_sci_roll{i}.fits')
        c._write_rc_file(p, roll, f'000010010010010010{i}', rng, planet=True,
                         star_xy=np.add(c.STAR_XY, c.SCI_OFFSETS[i]), disk=lambda roll, xy: detector_ring(roll, xy))
        sci.append(p)
    for i in range(c.NREF):
        p = str(d / f'rc_ref_{i}.fits')
        c._write_rc_file(p, 0.0, f'000010010010020010{i}', rng, psfref=True,
                         star_xy=np.add(c.STAR_XY, c.REF_OFFSETS[i]))
        ref.append(p)
    return sci, ref, str(d)


@pytest.fixture(scope='module')
def ring_rdi(ring_data):
    sci, ref, d = ring_data
    db = RCDatabase(d + '/out')
    db.add_concat(sci, ref)
    rdi = RCRDI(db, pad_data='auto', verbose=False)
    rdi.load_concat(0)
    rdi.prepare_convolution(osamp=OSAMP, fov_pixels=FOV, grid_kwargs=dict(fwhm=FWHM))
    return rdi


def _fit_region(rdi, im):
    y, x = np.indices(im.shape)
    r = np.hypot(x - rdi.c_star[0], y - rdi.c_star[1])
    pa = np.deg2rad(c.PLANET_PA_DEG)
    xp = rdi.c_star + c.PLANET_SEP_PX*np.array([-np.sin(pa), np.cos(pa)])
    return (r > 6) & (r < 24) & (np.hypot(x - xp[0], y - xp[1]) > PLANET_MASK) & np.isfinite(im)


def test_ring_fm_flux(ring_rdi):
    # Convolving the north-up raw model reproduces the ring as written into each roll: flux and shape.
    rdi = ring_rdi
    model, cm = raw_ring_model()
    cube = rdi.convolver.convolve_model(model, pxscale_in=rdi.pxscale/SUB, c_star_in=cm)
    for im, roll in zip(cube, rdi._posangs_sci):
        truth = detector_ring(roll, rdi.c_star, im.shape)
        assert np.isclose(im.sum(), model.sum()/SUB**2, rtol=0.01)
        assert np.isclose(im.sum(), truth.sum(), rtol=0.01)
        assert np.abs(im - truth).max() < 0.03*truth.max()


def _data_and_fm(rdi):
    data = rdi.run_rdi()
    model, cm = raw_ring_model()
    rdi.set_circumstellar_model(raw_model=model, raw_model_pxscale=rdi.pxscale/SUB, raw_model_center=cm)
    fm = rdi.run_rdi(forward_model=True)
    return data, fm


def test_amplitude_fit(ring_rdi):
    # Linear least-squares amplitude of the FM recovers the injected ring scale.
    data, fm = _data_and_fm(ring_rdi)
    m = _fit_region(ring_rdi, data.im)
    amp = np.sum(data.im[m]*fm.im[m]) / np.sum(fm.im[m]**2)
    assert np.isclose(amp, 1.0, rtol=0.02)


def test_ring_fm_chi2(ring_rdi):
    # Residual of data - FM is consistent with the propagated noise; a wrong amplitude or no ring is rejected.
    data, fm = _data_and_fm(ring_rdi)
    m = _fit_region(ring_rdi, data.im) & np.isfinite(data.err)

    def chi2(scale):
        return np.sum(((data.im - scale*fm.im)[m] / data.err[m])**2) / (m.sum() - 1)

    assert 0.5 < chi2(1.0) < 1.5
    assert chi2(0.9) > 2*chi2(1.0)
    assert chi2(0.0) > 100*chi2(1.0)


def test_known_model_autoscale(ring_rdi):
    # obj_fn with no fitted parameters and the convolved known model as a fixed component: autoscale returns its
    # amplitude (the tutorial 8 pattern).
    import lmfit
    from winnie.modeling import make_region_of_interest_mask, obj_fn
    rdi = ring_rdi
    rdi.rdi_presets()
    data = rdi.run_rdi(derotate=False)
    model, cm = raw_ring_model()
    cube = rdi.convolver.convolve_model(model, pxscale_in=rdi.pxscale/SUB, c_star_in=cm)
    rdi.set_circumstellar_model(model_cube=cube)
    fm = rdi.run_rdi(forward_model=True, derotate=False)
    pa = np.deg2rad(c.PLANET_PA_DEG)
    roi = make_region_of_interest_mask(data, rdi, rmin=6, rmax=24, rexcl_northup=PLANET_MASK,
                                       dxy_northup=[c.PLANET_SEP_PX*np.array([-np.sin(pa), np.cos(pa)])])
    out = obj_fn(lmfit.Parameters(), data, rdi, roi, 1., modelcubes_in=[cube], modellabels_in=['disk'],
                 fmreducs_in=[fm], autoscale_in=[True], return_soln=True, count=False, clear_each_call=False)
    coeffs = out[0]
    assert np.isclose(coeffs[0], 1.0, rtol=0.03)
    res = obj_fn(lmfit.Parameters(), data, rdi, roi, 1., modelcubes_in=[cube], modellabels_in=['disk'],
                 fmreducs_in=[fm], autoscale_in=[True], err_weighting=True, count=False, clear_each_call=False)
    assert 0.5 < np.mean(res**2) < 1.5

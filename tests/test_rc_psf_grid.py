import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import conftest as c
from winnie import RCRDI
from winnie.roman import (RCDatabase, RCInstrument, generate_rc_psf_grid, rc_grid_offsets, rc_psf_grid_key,
                          rc_psf_grid_samples, rc_transmission_map)

RES_MAS = 50.0      # lambda/D of the fake optics (mas)
PX_MAS = 20.0       # fake "corgisim" pixel (mas); the data use c.PXSCALE
SCALE = 1.02        # fake plate-scale error: sources land at SCALE * nominal offset
TILT = np.array([0.1, -0.05])  # fake common tilt (px)
SIG = 1.2           # fake PSF sigma (px)
RVALS = (0., 1., 2., 3., 4., 6.)
THVALS = (0., 90., 180., 270.)
OSAMP, FOV = 2, 15


def _throughput(r_lamd):
    return 1 - np.exp(-r_lamd**2/2)


def _fake_optics(**kw):
    okw = {'cor_type': 'fake', 'use_errors': 1, 'dm1_v': np.zeros(3), **kw}
    return SimpleNamespace(optics_keywords=okw, cgi_mode='excam', cor_type='fake', bandpass='1', lam_um=[0.575],
                           res_mas=RES_MAS)


def fake_image(optics, dx, dy, nfine, sfine, weights, use_fpm=1, use_field_stop=1, use_errors=1, **kw):
    """Gaussian at SCALE*(dx, dy) + TILT (px), axis at nfine//2, attenuated by 1 - exp(-r^2/2) when use_fpm."""
    fake_image.calls += 1
    if getattr(fake_image, 'fail_after', None) is not None and fake_image.calls > fake_image.fail_after:
        raise RuntimeError('interrupted')
    tilt = TILT if use_errors else 0.
    x0, y0 = (SCALE*np.array([dx, dy])/PX_MAS + tilt)*sfine + nfine//2
    y, x = np.indices((nfine, nfine))
    im = np.exp(-((x - x0)**2 + (y - y0)**2)/(2*(SIG*sfine)**2))
    im *= 2.0/im.sum()
    if use_fpm:
        im *= _throughput(np.hypot(dx, dy)/RES_MAS)
    return im


fake_image.calls = 0


def _grid(tmp_path, **kw):
    fake_image.calls, fake_image.fail_after = 0, None
    args = dict(rvals_lamd=RVALS, thvals_deg=THVALS, osamp=OSAMP, fov_pixels=FOV, cache_dir=str(tmp_path),
                show_progress=False, sample_fn=fake_image)
    return rc_psf_grid_samples(_fake_optics(), **{**args, **kw})


def _centroid(im):
    y, x = np.indices(im.shape)
    return np.array([np.sum(x*im), np.sum(y*im)])/im.sum()


def test_grid_format(tmp_path):
    grid = _grid(tmp_path)
    n = len(grid['psfs'])
    assert n == 1 + (len(RVALS) - 1)*len(THVALS)
    assert grid['psfs'].shape[1:] == (FOV*OSAMP + 1, FOV*OSAMP + 1)
    assert grid['psfs'].dtype == np.float32
    lit = grid['throughput'] > 0
    assert np.allclose(grid['psfs'][lit].sum(axis=(1, 2)), 1, atol=1e-5)
    assert np.all(grid['psfs'][~lit] == 0)  # fake r=0 sample is fully occulted
    assert np.sum(grid['rvals_lamd'] == 0) == 1
    # Measured offset map: SCALE/PX_MAS px per mas, tilt as offset.
    assert np.allclose(grid['offset_map_A'], np.eye(2)*SCALE/PX_MAS, rtol=1e-3, atol=1e-6)
    assert np.allclose(grid['offset_map_b'], TILT, atol=2e-3)
    # Every PSF is centered on where its source landed (subpixel shift included).
    cen = (FOV*OSAMP)/2
    for psf in grid['psfs'][lit]:
        assert np.all(np.abs(_centroid(psf) - cen) < 0.01*OSAMP)


def test_throughput(tmp_path):
    grid = _grid(tmp_path)
    assert np.allclose(grid['throughput'], _throughput(grid['rvals_lamd']), rtol=1e-3, atol=1e-6)


def test_offsets(tmp_path):
    grid = _grid(tmp_path)
    polar, xy = rc_grid_offsets(grid, RES_MAS, c.PXSCALE)
    # Radii in the data frame include the measured plate-scale error.
    assert np.allclose(polar[0], grid['rvals_lamd']*RES_MAS*SCALE/PX_MAS*c.PXSCALE, rtol=1e-3)
    assert np.allclose(np.hypot(*xy), polar[0])
    i = np.where((grid['thvals_deg'] == 90) & (grid['rvals_lamd'] > 0))[0][0]
    assert xy[0, i] < 0 and abs(xy[1, i]) < 1e-3*abs(xy[0, i])  # PA 90 deg points to -x


def test_cache_and_key(tmp_path):
    grid = _grid(tmp_path)
    fake_image.calls = 0
    again = _grid(tmp_path)
    assert fake_image.calls == 0
    assert np.array_equal(again['psfs'], grid['psfs'])
    key = grid['key']
    assert rc_psf_grid_key(_fake_optics(), RVALS, THVALS, OSAMP, FOV, sample_fn=fake_image) == key
    assert rc_psf_grid_key(_fake_optics(), RVALS[:-1], THVALS, OSAMP, FOV, sample_fn=fake_image) != key
    assert rc_psf_grid_key(_fake_optics(), RVALS, THVALS, 3, FOV, sample_fn=fake_image) != key
    assert rc_psf_grid_key(_fake_optics(dm1_v=np.ones(3)), RVALS, THVALS, OSAMP, FOV, sample_fn=fake_image) != key
    # Output-size keywords don't change the grid.
    assert rc_psf_grid_key(_fake_optics(output_dim=99), RVALS, THVALS, OSAMP, FOV, sample_fn=fake_image) == key
    _grid(tmp_path, rvals_lamd=RVALS[:-1])
    assert fake_image.calls > 0


def test_resume(tmp_path):
    n = 1 + (len(RVALS) - 1)*len(THVALS)
    fake_image.calls, fake_image.fail_after = 0, 5 + 5  # 5 calibration images, then 5 samples
    with pytest.raises(RuntimeError):
        rc_psf_grid_samples(_fake_optics(), RVALS, THVALS, osamp=OSAMP, fov_pixels=FOV, cache_dir=str(tmp_path),
                            show_progress=False, sample_fn=fake_image)
    fake_image.calls, fake_image.fail_after = 0, None
    grid = rc_psf_grid_samples(_fake_optics(), RVALS, THVALS, osamp=OSAMP, fov_pixels=FOV,
                               cache_dir=str(tmp_path), show_progress=False, sample_fn=fake_image)
    assert fake_image.calls == n - 5
    assert np.allclose(grid['psfs'], _grid(tmp_path / 'fresh')['psfs'])


def test_transmission_map(tmp_path):
    inst = RCInstrument(c.PXSCALE, 0.575)
    kw = dict(optics=_fake_optics(), rvals_lamd=RVALS, thvals_deg=THVALS, fov_pixels=FOV, cache_dir=str(tmp_path),
              show_progress=False, sample_fn=fake_image)
    _grid(tmp_path)
    shape, cc = (61, 61), np.array([30., 30.])
    tmap = rc_transmission_map(inst, cc, osamp=OSAMP, shape=shape, **kw)
    assert tmap.shape == (shape[0]*OSAMP, shape[1]*OSAMP)
    px_per_lamd = RES_MAS*SCALE/PX_MAS
    for r_lamd, pa in [(2, 0), (3, 90), (2.5, 0), (3.5, 180)]:
        t = np.deg2rad(pa)
        x, y = cc + r_lamd*px_per_lamd*np.array([-np.sin(t), np.cos(t)])
        val = tmap[int(round(y*OSAMP + 0.5*(OSAMP - 1))), int(round(x*OSAMP + 0.5*(OSAMP - 1)))]
        # Linear interpolation between radial samples 1 lambda/D apart.
        assert abs(val - _throughput(r_lamd)) < 0.06
    assert np.isclose(tmap[0, 0], _throughput(6), atol=1e-3)  # outside the grid: nearest sample
    # Derotation by 180 deg about a star 5 px from the mask center moves the mask center 10 px.
    rot = rc_transmission_map(inst, cc, osamp=OSAMP, shape=shape, posang=180, c_star=cc + [5, 0], **kw)
    iy, ix = np.unravel_index(np.argmin(rot), rot.shape)
    assert np.hypot(*(np.array([ix, iy])/OSAMP - 0.25 - (cc + [10, 0]))) < 0.5
    small = rc_transmission_map(inst, cc, osamp=OSAMP, shape=shape, return_oversample=False, **kw)
    assert small.shape == shape


@pytest.fixture
def rdi(rc_fixture, tmp_path):
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat(rc_fixture['sci'], rc_fixture['ref'])
    rdi = RCRDI(db, pad_data='auto', verbose=False)
    rdi.load_concat(0)
    return rdi


def _prepare(rdi, tmp_path, **kw):
    grid_kwargs = dict(optics=_fake_optics(), rvals_lamd=RVALS, thvals_deg=THVALS, cache_dir=str(tmp_path / 'grid'),
                       sample_fn=fake_image, **kw)
    rdi.prepare_convolution(osamp=OSAMP, fov_pixels=FOV, grid_fn=generate_rc_psf_grid, grid_kwargs=grid_kwargs)


def test_rc_convolver_wiring(rdi, tmp_path):
    _prepare(rdi, tmp_path)
    conv = rdi.convolver
    assert conv.transmission_map_fn is rc_transmission_map
    key = rc_psf_grid_key(_fake_optics(), RVALS, THVALS, OSAMP, FOV, sample_fn=fake_image)
    assert key[:8] in conv.psf_file
    assert conv.psfs.shape[1:] == (FOV*OSAMP + 1, FOV*OSAMP + 1)
    assert conv.coron_tmaps_osamp.min() < 0.1 and conv.coron_tmaps_osamp.max() > 0.99
    # A point source at 3 lambda/D: convolved flux = transmission there.
    r_lamd = 3.
    sep = r_lamd*RES_MAS*SCALE/PX_MAS
    n, sig = 61, 0.5
    c0 = (n - 1)/2
    y, x = np.indices((n, n))
    model = np.exp(-((x - c0)**2 + (y - c0 - sep)**2)/(2*sig**2))
    model /= model.sum()
    cube = conv.convolve_model(model, pxscale_in=rdi.pxscale, c_star_in=np.array([c0, c0]))
    for im, roll in zip(cube, rdi._posangs_sci):
        assert np.isclose(im.sum(), _throughput(r_lamd), rtol=0.03)
    # A different grid gets a different per-reduction cache file.
    old = conv.psf_file
    rdi.convolver = None
    _prepare(rdi, tmp_path, subsample=5)
    assert rdi.convolver.psf_file != old


def test_pickle(rdi, tmp_path):
    _prepare(rdi, tmp_path)
    conv = pickle.loads(pickle.dumps(rdi.convolver))
    assert np.array_equal(conv.psfs, rdi.convolver.psfs)
    assert np.array_equal(conv.coron_tmaps_osamp, rdi.convolver.coron_tmaps_osamp)


def test_grid_file(rdi, tmp_path):
    # A saved grid file gives the same grid, offsets and transmission map as the optics, without corgisim.
    grid = _grid(tmp_path)
    path = str(tmp_path / f"rc_psfgrid_{grid['key']}.fits")
    inst = RCInstrument(c.PXSCALE, 0.575)
    kw = dict(optics=_fake_optics(), rvals_lamd=RVALS, thvals_deg=THVALS, fov_pixels=FOV, osamp=OSAMP,
              cache_dir=str(tmp_path), show_progress=False, sample_fn=fake_image)
    a = generate_rc_psf_grid(inst, **kw)
    b = generate_rc_psf_grid(inst, osamp=OSAMP, fov_pixels=FOV, grid_file=path)
    for x, y in zip(a, b):
        assert np.allclose(x, y)
    shape, cc = (41, 41), np.array([20., 20.])
    ta = rc_transmission_map(inst, cc, shape=shape, posang=30, **{k: v for k, v in kw.items() if k != 'osamp'})
    tb = rc_transmission_map(inst, cc, shape=shape, posang=30, fov_pixels=FOV, grid_file=path)
    assert np.allclose(ta, tb)
    rdi.prepare_convolution(osamp=OSAMP, fov_pixels=FOV, grid_fn=generate_rc_psf_grid,
                            grid_kwargs=dict(grid_file=path))
    assert grid['key'][:8] in rdi.convolver.psf_file
    assert rdi.convolver.transmission_map_fn is rc_transmission_map
    assert np.allclose(rdi.convolver.psfs, grid['psfs'])
    with pytest.raises(ValueError):
        generate_rc_psf_grid(inst)


def test_deconvolution(rdi, tmp_path):
    # No SIAF aperture: the transmission map must not go through the JWST distortion model.
    _prepare(rdi, tmp_path)
    assert rdi.convolver.transmission_map_fn is rc_transmission_map
    deconv, reduc = rdi.run_deconvolution(num_iter=2, show_progress=False)
    assert deconv.im.shape == reduc.im.shape
    assert np.isfinite(deconv.im).mean() > 0.5

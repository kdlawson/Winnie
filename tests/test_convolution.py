import numpy as np
import pytest
import torch
from astropy.convolution import Gaussian2DKernel
from scipy import signal

from conftest import assert_close
from winnie import backend, convolution, deconvolution
from winnie.convolution import SpatialPSFConvolver, _get_psf_patch_bounds


def make_problem(ny=60, nx=70, psf_n=15, seed=0):
    """A scene, a stack of 7 PSFs, a map of which PSF to use for each pixel, and a transmission map."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((ny, nx))
    cy, cx = (ny - 1) / 2 + 0.3, (nx - 1) / 2 - 1.1
    r = np.hypot(xx - cx, yy - cy)
    th = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
    ring, sector = np.digitize(r, [6, 20]), (th // 120).astype(int)
    psf_inds = np.where(ring == 0, 0, 1 + (ring - 1)*3 + sector).astype(np.int32)
    py, px = np.indices((psf_n, psf_n)) - (psf_n - 1)/2
    psfs = np.stack([np.exp(-0.5*((px - 0.3*np.sin(k))**2 + (py - 0.2*np.cos(k))**2)/(1.5 + 0.2*k)**2) for k in range(7)]).astype(np.float32)
    psfs /= psfs.sum(axis=(1, 2), keepdims=True)
    scene = np.exp(-0.5*((r - 20)/3)**2) + 0.05*rng.random((ny, nx))
    tmap = 0.5 + 0.5*np.cos(xx/9.)*np.sin(yy/7.)
    return scene, psfs, psf_inds, tmap


def reference_convolution(im, psfs, psf_inds, coron_tmap=None):
    im = im if coron_tmap is None else im*coron_tmap
    out = np.zeros(im.shape)
    for i in np.unique(psf_inds):
        out += signal.fftconvolve(np.where(psf_inds == i, im, 0.), psfs[i].astype(np.float64), mode='same')
    return out


@pytest.mark.parametrize('method', ['full', 'localized', 'auto'])
@pytest.mark.parametrize('psf_n', [15, 14]) # odd and even PSF sizes
def test_spatial_convolution_matches_scipy(device, rtol, method, psf_n):
    scene, psfs, psf_inds, tmap = make_problem(psf_n=psf_n)
    out = convolution.convolve_with_spatial_psfs(scene, psfs, psf_inds, coron_tmap=tmap, method=method, device=device)
    assert out.dtype == scene.dtype
    assert_close(out, reference_convolution(scene, psfs, psf_inds, tmap), rtol)


@pytest.mark.parametrize('method', ['full', 'localized'])
def test_spatial_convolution_with_psfs_larger_than_image(device, rtol, method):
    scene, psfs, psf_inds, _ = make_problem(psf_n=41)
    scene, psf_inds = scene[:20, :23], psf_inds[:20, :23]
    out = convolution.convolve_with_spatial_psfs(scene, psfs, psf_inds, method=method, device=device)
    assert_close(out, reference_convolution(scene, psfs, psf_inds), rtol)


@pytest.mark.parametrize('method', ['full', 'localized'])
def test_spatial_convolution_with_a_subset_of_psfs(device, rtol, method):
    scene, psfs, psf_inds, _ = make_problem()
    psf_inds = np.where(psf_inds % 2 == 0, psf_inds, 0).astype(np.int32) # PSFs 1, 3, and 5 are never used
    used_inds, patch_bounds = _get_psf_patch_bounds(psf_inds, len(psfs), (8, 8))
    expected = reference_convolution(scene, psfs, psf_inds)
    for kwargs in [{}, dict(used_inds=used_inds, patch_bounds=patch_bounds)]:
        out = convolution.convolve_with_spatial_psfs(scene, psfs, psf_inds, method=method, device=device, **kwargs)
        assert_close(out, expected, rtol)


def test_spatial_convolution_preserves_float32_images(device):
    scene, psfs, psf_inds, _ = make_problem()
    out = convolution.convolve_with_spatial_psfs(scene.astype(np.float32), psfs, psf_inds, device=device)
    assert out.dtype == np.float32
    assert_close(out, reference_convolution(scene, psfs, psf_inds), 5e-5)


@pytest.mark.parametrize('psf_n', [15, 14])
@pytest.mark.parametrize('kernel_type', ['array', 'astropy'])
def test_psf_convolve_matches_scipy(device, rtol, psf_n, kernel_type):
    scene, psfs, _, _ = make_problem(psf_n=psf_n)
    kernel = psfs[2] if kernel_type == 'array' else Gaussian2DKernel(1.5, 2., 0.3)
    expected = signal.fftconvolve(scene, np.asarray(kernel, dtype=np.float64), mode='same')
    assert_close(convolution.psf_convolve(scene, kernel, device=device), expected, rtol)


def test_convolver_caches_psf_ffts_and_batches(device, rtol):
    scene, psfs, psf_inds, tmap = make_problem()
    expected = reference_convolution(scene, psfs, psf_inds, tmap)

    convolver = SpatialPSFConvolver(psfs, device=device)
    for _ in range(2): # the second call uses the cached transforms
        assert_close(convolver(scene, psf_inds, coron_tmap=tmap, method='full'), expected, rtol)
    assert len(convolver._full_ffts) == 1 and convolver._cache_bytes > 0

    one_psf_per_batch = SpatialPSFConvolver(psfs, device=device, max_batch_bytes=1)
    assert_close(one_psf_per_batch(scene, psf_inds, coron_tmap=tmap, method='full'), expected, rtol)

    uncached = SpatialPSFConvolver(psfs, device=device, max_cache_bytes=0)
    for method in ['full', 'localized']:
        assert_close(uncached(scene, psf_inds, coron_tmap=tmap, method=method), expected, rtol)
    assert uncached._cache_bytes == 0


def test_convolver_accepts_and_returns_tensors(device):
    scene, psfs, psf_inds, _ = make_problem()
    convolver = SpatialPSFConvolver(psfs, device=device)
    dtype = backend.resolve_dtype(device, np.float64)
    out = convolver(backend.to_tensor(scene, device, dtype), backend.to_tensor(psf_inds, device))
    assert isinstance(out, torch.Tensor) and out.device.type == torch.device(device).type
    assert_close(backend.to_numpy(out), convolver(scene, psf_inds), 5e-5)


def test_localized_convolution_is_the_same_in_parallel_and_in_serial():
    scene, psfs, psf_inds, _ = make_problem()
    n_threads = torch.get_num_threads()
    try:
        results = {}
        for threads in [1, 4]:
            torch.set_num_threads(threads)
            results[threads] = SpatialPSFConvolver(psfs, device='cpu')(scene, psf_inds, method='localized')
    finally:
        torch.set_num_threads(n_threads)
    np.testing.assert_array_equal(results[1], results[4])


def test_convolver_argument_checks():
    scene, psfs, psf_inds, _ = make_problem()
    with pytest.raises(ValueError, match='psf_inds'):
        SpatialPSFConvolver(psfs, device='cpu')(scene)
    with pytest.raises(ValueError, match='method'):
        SpatialPSFConvolver(psfs, device='cpu')(scene, psf_inds, method='fast')


def reference_richardson_lucy(image, psfs, psf_inds, im_mask, num_iter, epsilon, excl_mask):
    """The algorithm of winnie.deconvolution.coronagraphic_richardson_lucy (with mirror_psfs=True), using scipy."""
    psfs_inv = psfs[..., ::-1, ::-1]
    im_deconv = np.full(image.shape, np.nanmedian(image[image > 0]))
    unity_conv = reference_convolution(np.ones_like(image), psfs_inv, psf_inds)
    for i in range(num_iter):
        conv = reference_convolution(im_deconv*im_mask, psfs, psf_inds)
        conv += np.sign(conv)*1e-12
        relative_blur = np.where(np.abs(conv) < epsilon, 0, image/conv) if i > 0 else image/conv
        fcorr = np.where(excl_mask, 1, reference_convolution(relative_blur, psfs_inv, psf_inds)/unity_conv)
        im_deconv *= fcorr
    return im_deconv


@pytest.mark.parametrize('method', ['full', 'localized'])
def test_richardson_lucy_matches_reference(device, rtol, method):
    scene, psfs, psf_inds, tmap = make_problem()
    image = reference_convolution(scene, psfs, psf_inds, tmap) + 1e-3
    excl_mask = np.zeros(image.shape, dtype=bool)
    excl_mask[:4] = True
    kwargs = dict(psf_inds=psf_inds, im_mask=tmap, epsilon=1e-2, excl_mask=excl_mask, mirror_psfs=True,
                  show_progress=False, convolution_method=method, device=device)
    out = deconvolution.coronagraphic_richardson_lucy(image, psfs, num_iter=6, **kwargs)
    assert out.dtype == image.dtype
    assert_close(out, reference_richardson_lucy(image, psfs, psf_inds, tmap, 6, 1e-2, excl_mask), rtol*10)


def test_richardson_lucy_returns_requested_iterations(device, rtol):
    scene, psfs, psf_inds, tmap = make_problem()
    image = reference_convolution(scene, psfs, psf_inds, tmap) + 1e-3
    kwargs = dict(psf_inds=psf_inds, im_mask=tmap, show_progress=False, device=device)
    final, iters = deconvolution.coronagraphic_richardson_lucy(image, psfs, num_iter=5, return_iters=np.array([0, 3]), **kwargs)
    assert iters.shape == (2, *image.shape)
    for i, num_iter in zip([0, 1], [1, 4]):
        assert_close(iters[i], deconvolution.coronagraphic_richardson_lucy(image, psfs, num_iter=num_iter, **kwargs), rtol)
    assert_close(final, deconvolution.coronagraphic_richardson_lucy(image, psfs, num_iter=5, **kwargs), rtol)


@pytest.mark.parametrize('mask_dtype', [np.uint8, np.int64, np.float64])
def test_richardson_lucy_accepts_exclusion_masks_that_are_not_boolean(device, mask_dtype):
    scene, psfs, psf_inds, tmap = make_problem()
    image = reference_convolution(scene, psfs, psf_inds, tmap) + 1e-3
    excl_mask = np.zeros(image.shape, dtype=bool)
    excl_mask[:4] = True
    kwargs = dict(psf_inds=psf_inds, im_mask=tmap, num_iter=3, show_progress=False, device=device)
    expected = deconvolution.coronagraphic_richardson_lucy(image, psfs, excl_mask=excl_mask, **kwargs)
    out = deconvolution.coronagraphic_richardson_lucy(image, psfs, excl_mask=excl_mask.astype(mask_dtype), **kwargs)
    np.testing.assert_array_equal(out, expected)


def test_stpsf_model_center_offset_is_the_same_on_every_device(device):
    yy, xx = np.indices((140, 140))
    psf_off = np.exp(-0.5*((xx - 70.1)**2 + (yy - 69.4)**2)/4.**2)
    expected = convolution.get_stpsf_model_center_offset(psf_off, 4, device='cpu')
    np.testing.assert_allclose(convolution.get_stpsf_model_center_offset(psf_off, 4, device=device), expected, atol=1e-3)


def test_richardson_lucy_with_a_single_psf(device, rtol):
    scene, psfs, psf_inds, _ = make_problem()
    image = signal.fftconvolve(scene, psfs[3].astype(np.float64), mode='same') + 1e-3
    out = deconvolution.coronagraphic_richardson_lucy(image, psfs[3], num_iter=4, mirror_psfs=True, show_progress=False, device=device)
    expected = reference_richardson_lucy(image, np.stack([psfs[3]]), np.zeros(image.shape, dtype=int), 1., 4, 0, np.zeros(image.shape, dtype=bool))
    assert_close(out, expected, rtol*10)

import numpy as np
import pytest
from scipy import ndimage

from conftest import assert_close
from winnie import utils


def smooth_image(ny=50, nx=56, cy=24.3, cx=27.6, sigma=5.):
    yy, xx = np.indices((ny, nx))
    return np.exp(-0.5*((xx - cx)**2 + (yy - cy)**2)/sigma**2) + 0.1 + 0.02*np.sin(xx/4.)*np.cos(yy/5.)


def test_rotation_by_90_degrees_matches_rot90(device, rtol):
    im = smooth_image(51, 51)
    assert_close(utils.rotate_image(im, 90., device=device), np.rot90(im), rtol)
    assert_close(utils.rotate_image(im, -90., device=device), np.rot90(im, -1), rtol)


def test_rotation_follows_the_convention_of_xy_polar_ang_displacement(device):
    im = smooth_image()
    cent = np.array([20.4, 30.7])
    xg, yg = np.meshgrid(np.arange(im.shape[1], dtype=float), np.arange(im.shape[0], dtype=float))
    x, y = utils.xy_polar_ang_displacement(xg - cent[0], yg - cent[1], 33.3)
    expected = ndimage.map_coordinates(im, [y + cent[1], x + cent[0]], order=3, mode='constant', cval=0.)
    out = utils.rotate_image(im, 33.3, cent=cent, cval0=0., device=device)
    # bicubic and spline interpolation differ slightly, but a wrong direction or center would differ by order unity
    assert np.max(np.abs(out - expected)) < 1e-2*im.max()


def test_big_endian_float64_images_are_processed_like_native_ones(device):
    im = smooth_image()
    native = utils.rotate_image(im, 12.3, device=device)
    big_endian = utils.rotate_image(im.astype('>f8'), 12.3, device=device) # e.g., data as read from a FITS file
    np.testing.assert_array_equal(big_endian, native)


def test_rotation_keeps_the_center_fixed(device):
    yy, xx = np.indices((50, 60))
    im = np.exp(-0.5*((xx - 30.)**2 + (yy - 20.)**2)/3.**2)
    out = utils.rotate_image(im, 40., cent=(30., 20.), cval0=0., device=device)
    assert np.unravel_index(np.argmax(out), out.shape) == (20, 30)
    out = utils.rotate_image(im, 40., cent=(30., 20.), new_cent=(10., 15.), cval0=0., device=device)
    assert np.unravel_index(np.argmax(out), out.shape) == (15, 10)


def test_rotation_fills_outside_pixels_with_cval(device):
    im = np.ones((41, 41))
    out = utils.rotate_image(im, 45., cval0=np.nan, device=device)
    assert np.isnan(out[0, 0]) and np.isfinite(out[20, 20])
    out = utils.rotate_image(im, 45., cval0=0., device=device)
    assert out[0, 0] == 0. and out[20, 20] == pytest.approx(1.)


def test_rotation_propagates_nans_locally(device):
    im = np.ones((41, 41))
    im[20, 20] = np.nan
    out = utils.rotate_image(im, 25., cval0=0., device=device)
    nan_pixels = np.argwhere(np.isnan(out))
    assert 1 <= len(nan_pixels) <= 9
    assert np.all(np.hypot(*(nan_pixels - 20).T) < 2)


def test_rotation_does_not_mix_zeros_into_neighbors(device):
    im = np.ones((41, 41))
    im[15:26, 15:26] = 0. # invalid pixels, which are given the exact value of zero
    out = utils.rotate_image(im, 20., cval0=np.nan, device=device)
    assert out[20, 20] == 0. and out[5, 20] == pytest.approx(1.)
    assert abs(np.sum(out == 0.) - 121) < 40 # the zeros stay as a block with about the same area
    assert np.nanmin(out[out != 0.]) > 0.5 # and none of them are mixed into the neighboring pixels


def test_zero_angles_return_exact_copies(device):
    im = smooth_image()
    out = utils.rotate_image(im, 0., device=device)
    assert out is not im and np.array_equal(out, im)
    hcube = np.stack([im, 2*im, 3*im]).astype(np.float64)
    out = utils.rotate_hypercube(hcube, [30., 0., -30.], device=device)
    np.testing.assert_array_equal(out[1], hcube[1]) # even when the calculations are done in float32
    assert not np.array_equal(out[0], hcube[0])


def test_rotate_hypercube_matches_rotating_each_image(device, rtol, monkeypatch):
    rng = np.random.default_rng(0)
    hcube = smooth_image(30, 34) + rng.random((3, 2, 30, 34))*0.1
    hcube[1, 0, 5:8, 6:9] = np.nan
    angles, cent = np.array([10., -20., 40.]), (14.3, 15.1)
    expected = np.stack([utils.rotate_image(h, a, cent=cent, cval0=np.nan, device=device) for h, a in zip(hcube, angles)])
    assert_close(utils.rotate_hypercube(hcube, angles, cent=cent, cval0=np.nan, device=device), expected, rtol)
    monkeypatch.setattr(utils, '_MAX_CHUNK_BYTES', 1) # one exposure at a time
    assert_close(utils.rotate_hypercube(hcube, angles, cent=cent, cval0=np.nan, device=device), expected, rtol)


def test_pad_and_rotate_hypercube_avoids_losing_pixels(device):
    hcube = np.ones((2, 30, 34))
    angles, cent = np.array([40., -40.]), np.array([14.3, 15.1])
    out, cent_pad = utils.pad_and_rotate_hypercube(hcube, angles, cent=cent, device=device)
    dymin, dymax, dxmin, dxmax = utils.compute_derot_padding(34, 30, angles, cent=cent)
    assert out.shape == (2, 30 + dymin + dymax, 34 + dxmin + dxmax)
    np.testing.assert_array_equal(cent_pad, cent + [dxmin, dymin])
    # Pixels along the edge of the rotated image are NaN (since they involve the NaN padding), but nearly all of the
    # flux is retained, unlike when rotating without padding
    unpadded = utils.rotate_hypercube(hcube, angles, cent=cent, cval0=np.nan, device=device)
    assert np.nansum(out[0]) > 0.9*hcube[0].sum()
    assert np.nansum(unpadded[0]) < 0.85*hcube[0].sum()


def reference_shift(im, new_shape, shift_xy, cval):
    """out[y, x] = im[y + shift_y, x + shift_x], with cval wherever that falls outside of im."""
    out = np.full(new_shape, cval, dtype=im.dtype)
    for y in range(new_shape[0]):
        for x in range(new_shape[1]):
            yi, xi = y + shift_xy[1], x + shift_xy[0]
            if 0 <= yi < im.shape[0] and 0 <= xi < im.shape[1]:
                out[y, x] = im[yi, xi]
    return out


@pytest.mark.parametrize('new_shape', [(50, 60), (20, 22)])
def test_pad_or_crop_with_integer_centers_is_an_exact_shift(device, new_shape):
    im = smooth_image(30, 34).astype(np.float32)
    im[3, 4] = np.nan
    out = utils.pad_or_crop_image(im, new_shape, cent=np.array([10, 12]), new_cent=np.array([20, 18]), cval0=-1., device=device)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, reference_shift(im, new_shape, (10 - 20, 12 - 18), -1.))


def test_pad_or_crop_to_the_same_geometry_returns_a_copy(device):
    im = smooth_image()
    out = utils.pad_or_crop_image(im, im.shape, device=device)
    assert out is not im and np.array_equal(out, im)


def fractional_shift_coordinates(shape_out, cent, new_cent):
    x, y = np.meshgrid(np.arange(shape_out[1]) + cent[0] - new_cent[0], np.arange(shape_out[0]) + cent[1] - new_cent[1])
    return x, y


def test_pad_or_crop_with_fractional_centers_matches_scipy_for_bilinear(device, rtol):
    im = smooth_image(30, 34)
    cent, new_cent, shape_out = np.array([10.4, 12.3]), np.array([20., 18.]), (40, 36)
    x, y = fractional_shift_coordinates(shape_out, cent, new_cent)
    expected = ndimage.map_coordinates(im, [y, x], order=1, mode='constant', cval=0.)
    assert_close(utils.pad_or_crop_image(im, shape_out, cent=cent, new_cent=new_cent, cval0=0., order=1, device=device), expected, rtol)


def test_pad_or_crop_with_high_orders_uses_scipy_splines(device):
    im = smooth_image(30, 34)
    cent, new_cent, shape_out = np.array([10.4, 12.3]), np.array([20., 18.]), (40, 36)
    x, y = fractional_shift_coordinates(shape_out, cent, new_cent)
    expected = ndimage.map_coordinates(im, [y, x], order=5, mode='constant', cval=0.)
    out = utils.pad_or_crop_image(im, shape_out, cent=cent, new_cent=new_cent, cval0=0., order=5, device=device)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12) # always on the CPU, and in float64


@pytest.mark.parametrize('bad_value', [np.nan, 0.])
def test_pad_or_crop_propagates_nans_and_zeros_over_the_kernel_footprint(device, bad_value):
    im = smooth_image(30, 34)
    qy, qx = 15, 17
    im[qy, qx] = bad_value
    cent, new_cent = np.array([10.4, 12.3]), np.array([10., 12.])
    out = utils.pad_or_crop_image(im, (30, 34), cent=cent, new_cent=new_cent, cval0=1., device=device)
    flagged = np.isnan(out) if np.isnan(bad_value) else (out == 0.)
    # An output pixel is flagged if any pixel in the 4x4 footprint of bicubic interpolation is bad, so for a single bad
    # pixel, the output samples at x in [qx-2, qx+2) and y in [qy-2, qy+2) (in input coordinates) are affected.
    x, y = fractional_shift_coordinates((30, 34), cent, new_cent)
    expected = (x >= qx - 2) & (x < qx + 2) & (y >= qy - 2) & (y < qy + 2)
    np.testing.assert_array_equal(flagged, expected)
    assert flagged.sum() == 16


def test_sample_image_is_aligned_with_the_pixel_grid(device):
    im = np.random.default_rng(1).random((20, 24))
    yy, xx = np.indices(im.shape).astype(float)
    for order in [0, 1, 3]:
        assert_close(utils.sample_image(im, xx, yy, order=order, device=device), im, 1e-6)
    out = utils.sample_image(im, xx[:5] + 30, yy[:5], order=1, cval=-7., device=device)
    assert np.all(out == -7.) # entirely outside of the image


def test_sample_image_matches_scipy_for_bilinear_interpolation(device, rtol):
    rng = np.random.default_rng(2)
    im = smooth_image(30, 34) + 0.1*rng.random((30, 34))
    x, y = rng.uniform(-3, 36, (2, 25, 28))
    # Samples within a rounding error of the edge of the image can legitimately be considered inside or outside.
    x, y = [np.where((np.abs(c) < 0.01) | (np.abs(c - (n - 1)) < 0.01), c + 0.05, c) for c, n in [(x, 34), (y, 30)]]
    expected = ndimage.map_coordinates(im, [y, x], order=1, mode='constant', cval=-1.)
    assert_close(utils.sample_image(im, x, y, order=1, cval=-1., device=device), expected, rtol)
    stack = np.stack([im, 2*im])
    out = utils.sample_image(stack, x, y, order=1, cval=-1., device=device) # the same coordinates apply to every image
    assert out.shape == (2, 25, 28)
    assert_close(out[0], expected, rtol)

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from conftest import assert_close
from winnie.distortion import _interp_image, _make_coord_grid


def make_image(shape=(30, 34)):
    yy, xx = np.indices(shape)
    return np.exp(-0.5*((xx - 15.3)**2 + (yy - 14.1)**2)/25.) + 0.1 + 0.02*np.sin(xx/4.)*np.cos(yy/5.)


@pytest.mark.parametrize('osamp', [1, 2])
@pytest.mark.parametrize('method', ['linear', 'nearest'])
def test_interpolation_matches_scipy(device, rtol, osamp, method):
    im = make_image((30*osamp, 34*osamp))
    x_grid, y_grid = _make_coord_grid(im.shape, osamp=osamp)
    rng = np.random.default_rng(osamp)
    x_samp = x_grid + rng.uniform(-3, 3, im.shape) # some samples will fall outside of the grid
    y_samp = y_grid + rng.uniform(-3, 3, im.shape)
    interpolator = RegularGridInterpolator((y_grid[:, 0], x_grid[0]), im, method=method, bounds_error=False, fill_value=-1.)
    expected = interpolator(np.array([y_samp.ravel(), x_samp.ravel()]).T).reshape(im.shape)
    out = _interp_image(im, x_grid, y_grid, x_samp, y_samp, method=method, fill_value=-1., prop_nans=False, device=device)
    if method == 'linear':
        assert_close(out, expected, rtol)
    else: # nearest neighbor choices can differ for samples (to within rounding) at the half pixel
        np.testing.assert_array_equal(out == -1., expected == -1.)
        assert np.sum(np.abs(out - expected) > 1e-6) <= 2


def test_cubic_interpolation_propagates_nans(device):
    im = make_image()
    im[10:14, 12:16] = np.nan
    x_grid, y_grid = _make_coord_grid(im.shape)
    rng = np.random.default_rng(3)
    x_samp, y_samp = x_grid + rng.uniform(-1, 1, im.shape), y_grid + rng.uniform(-1, 1, im.shape)
    out = _interp_image(im, x_grid, y_grid, x_samp, y_samp, method='cubic', fill_value=np.nan, prop_nans=True, prop_threshold=0.1, device=device)

    nans = np.isnan(im).astype(float) # NaN wherever the linearly interpolated NaN indicator exceeds the threshold
    indicator = RegularGridInterpolator((y_grid[:, 0], x_grid[0]), nans, method='linear', bounds_error=False, fill_value=0)
    expected_nans = indicator(np.array([y_samp.ravel(), x_samp.ravel()]).T).reshape(im.shape) > 0.1
    expected_nans |= (x_samp < 0) | (x_samp > im.shape[1] - 1) | (y_samp < 0) | (y_samp > im.shape[0] - 1) # fill_value
    assert np.sum(np.isnan(out) != expected_nans) <= 2 # (samples exactly at the threshold can differ in float32)
    assert np.isfinite(out[~expected_nans]).all()


def test_unknown_interpolation_methods_are_rejected():
    im = make_image()
    x_grid, y_grid = _make_coord_grid(im.shape)
    with pytest.raises(ValueError, match='method'):
        _interp_image(im, x_grid, y_grid, x_grid, y_grid, method='quintic')

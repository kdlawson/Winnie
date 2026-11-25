import numpy as np
import pytest

from conftest import assert_close
from winnie import rdi, utils


def make_data(nT=3, nref=8, nL=2, ny=40, nx=44, seed=1, dtype=np.float64):
    """Correlated science and reference images (a shared bright core plus random low-order modes and noise) and zones."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((ny, nx))
    cent = np.array([21.3, 19.4])
    r, th = np.hypot(xx - cent[0], yy - cent[1]), np.arctan2(yy - cent[1], xx - cent[0])
    base = 100./(1 + (r/4)**2) + 1.
    modes = np.stack([np.sin((k + 1)*th + 0.1*(k + 2)*r) for k in range(4)])

    def make(n):
        imgs = base*(1 + np.einsum('nlk,kyx->nlyx', rng.normal(0, 0.1, (n, nL, 4)), modes))
        return (imgs + rng.normal(0, 0.05, imgs.shape)).astype(dtype)

    optzones, subzones = rdi.build_annular_rdi_zones(nx, ny, cent, r_opt=[[0, 12], [12, 40]], r_sub=None)
    return make(nT), make(nref), optzones, subzones, cent


def reference_coefficients(hcube, hcube_ref, optzones, ref_mask=None):
    nT, nL = hcube.shape[:2]
    nref = len(hcube_ref)
    coeffs = np.zeros((len(optzones), nT, nref, nL))
    for Ri, zone in enumerate(optzones):
        cols = np.arange(nref) if ref_mask is None else np.flatnonzero(ref_mask[Ri])
        for l in range(nL):
            A = hcube_ref[cols, l][:, zone].T
            for t in range(nT):
                coeffs[Ri, t, cols, l] = np.linalg.lstsq(A, hcube[t, l][zone], rcond=None)[0]
    return coeffs


def reference_model(hcube_ref, coeffs, subzones, err_hcube_ref=None):
    """Broadcast-and-sum PSF model reconstruction (the approach used before it was replaced by a matrix product)."""
    nR, nT, nref, nL = coeffs.shape
    model = np.full((nT, nL, *hcube_ref.shape[-2:]), np.nan)
    err = np.full_like(model, np.nan)
    for Ri in range(nR):
        sub, c = subzones[Ri], coeffs[Ri][..., np.newaxis]
        model[..., sub] = np.sum(c*hcube_ref[:, :, sub], axis=1)
        if err_hcube_ref is not None:
            err[..., sub] = np.sqrt(np.sum((c*err_hcube_ref[:, :, sub])**2, axis=1))
    return model, err


@pytest.fixture
def ref_mask():
    mask = np.ones((2, 8), dtype=bool)
    mask[0, :3] = False
    mask[1, 6:] = False
    return mask


@pytest.mark.parametrize('use_ref_mask', [False, True])
def test_coefficients_match_least_squares(device, rtol, ref_mask, use_ref_mask):
    hcube, hcube_ref, optzones, _, _ = make_data()
    mask = ref_mask if use_ref_mask else None
    coeffs = rdi.compute_rdi_coefficients(hcube, hcube_ref, optzones, ref_mask=mask, device=device)
    assert_close(coeffs, reference_coefficients(hcube, hcube_ref, optzones, mask), rtol)
    if use_ref_mask:
        assert np.all(coeffs[0][:, :3] == 0) # excluded references get no weight


def test_reference_matrices_keep_the_precision_of_the_data():
    hcube, hcube_ref, optzones, _, _ = make_data(dtype=np.float32)
    for refs in [hcube_ref, hcube_ref.astype(np.float64)]:
        prepared = rdi._prepare_reference_matrices(refs, optzones)
        assert all(a.dtype == refs.dtype for a in [*prepared['Qs'], *prepared['Rs']])
    assert rdi.compute_rdi_coefficients(hcube, hcube_ref, optzones, device='cpu').dtype == np.float32


def test_reconstruct_psf_model_matches_broadcast_sum(device, rtol):
    hcube, hcube_ref, _, subzones, _ = make_data()
    err_ref = 0.1 + np.abs(hcube_ref)*0.01
    coeffs = np.random.default_rng(3).normal(0, 0.3, (2, 3, 8, 2))
    subzones = subzones.copy()
    subzones[:, :5] = False # pixels that are in no zone are left as NaN
    model, err = rdi.reconstruct_psf_model(hcube_ref, coeffs, subzones, err_hcube_ref=err_ref, device=device)
    expected_model, expected_err = reference_model(hcube_ref, coeffs, subzones, err_ref)
    assert np.isnan(model[..., :5, :]).all() and np.isfinite(model[..., 5:, :]).any()
    assert_close(model, expected_model, rtol)
    assert_close(err, expected_err, rtol)
    assert rdi.reconstruct_psf_model(hcube_ref, coeffs, subzones, device=device)[1] is None


@pytest.mark.parametrize('use_ref_mask', [False, True])
def test_rdi_residuals_match_reference(device, rtol, ref_mask, use_ref_mask):
    hcube, hcube_ref, optzones, subzones, _ = make_data()
    err_hcube, err_ref = 0.1 + np.abs(hcube)*0.01, 0.1 + np.abs(hcube_ref)*0.01
    mask = ref_mask if use_ref_mask else None
    coeffs = reference_coefficients(hcube, hcube_ref, optzones, mask)
    model, model_err = reference_model(hcube_ref, coeffs, subzones, err_ref)

    res, res_err, _ = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, ref_mask=mask, err_hcube=err_hcube, err_hcube_ref=err_ref, device=device)
    assert_close(res, hcube - model, rtol, scale=np.abs(hcube).max()) # the precision is set by that of the (much brighter) model
    assert_close(res_err, np.hypot(err_hcube, model_err), rtol)
    assert_close(rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, ref_mask=mask, return_psf_model=True, device=device), model, rtol)
    assert_close(rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, ref_mask=mask, return_coeffs=True, device=device), coeffs, rtol)


def test_residual_uncertainties_without_reference_uncertainties(device, rtol):
    hcube, hcube_ref, optzones, subzones, _ = make_data()
    err_hcube = 0.1 + np.abs(hcube)*0.01
    _, res_err, _ = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, err_hcube=err_hcube, device=device)
    assert_close(res_err, err_hcube, 0) # the PSF model is assumed to be noiseless


def test_cached_reference_matrices_give_the_same_residuals(device, ref_mask):
    hcube, hcube_ref, optzones, subzones, _ = make_data()
    prepared = rdi._prepare_reference_matrices(hcube_ref, optzones, ref_mask=ref_mask)
    expected = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, ref_mask=ref_mask, device=device)[0]
    out = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, ref_mask=ref_mask, prepared_refmats=prepared, device=device)[0]
    np.testing.assert_array_equal(out, expected)


def test_model_constrained_residuals(device, rtol):
    hcube, hcube_ref, optzones, subzones, _ = make_data()
    css = 5*np.exp(-0.5*((np.indices(hcube.shape[-2:])[1] - 30.)**2 + (np.indices(hcube.shape[-2:])[0] - 12.)**2)/20.)*np.ones_like(hcube)
    hcube_sci = hcube + css
    coeffs = reference_coefficients(hcube_sci - css, hcube_ref, optzones)
    model, _ = reference_model(hcube_ref, coeffs, subzones)
    kwargs = dict(hcube_css=css, device=device)
    scale = np.abs(hcube_sci).max()
    assert_close(rdi.rdi_residuals(hcube_sci, hcube_ref, optzones, subzones, objective=False, **kwargs)[0], hcube_sci - model, rtol, scale=scale)
    assert_close(rdi.rdi_residuals(hcube_sci, hcube_ref, optzones, subzones, objective=True, **kwargs)[0], (hcube_sci - css) - model, rtol, scale=scale)


def test_zero_nans(device):
    hcube, hcube_ref, optzones, subzones, _ = make_data()
    hcube[0, :, 10:13, 20:23] = np.nan
    hcube_ref[2, :, 8, 18] = np.nan
    kwargs = dict(return_coeffs=True, device=device)
    assert np.isnan(rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, **kwargs)).any() # NaNs spoil the fit by default
    coeffs = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, zero_nans=True, **kwargs)
    np.testing.assert_array_equal(coeffs, rdi.rdi_residuals(np.nan_to_num(hcube), np.nan_to_num(hcube_ref), optzones, subzones, **kwargs))

    # The NaNs only affect the fit, not the pixels of the science (or reference) images that are NaN themselves.
    res = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, zero_nans=True, device=device)[0]
    expected_nans = np.zeros(res.shape, dtype=bool)
    expected_nans[0, :, 10:13, 20:23] = True
    expected_nans[:, :, 8, 18] = True
    np.testing.assert_array_equal(np.isnan(res), expected_nans)


def test_derotation(device, rtol):
    hcube, hcube_ref, optzones, subzones, cent = make_data()
    posangs = np.array([-20., 0., 35.])
    err_hcube = 0.1 + np.abs(hcube)*0.01
    kwargs = dict(err_hcube=err_hcube, device=device)
    res, err, _ = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, **kwargs)
    derot, derot_err, cent_out = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, posangs=posangs, cent=cent, **kwargs)
    assert_close(derot, utils.rotate_hypercube(res, -posangs, cent=cent, cval0=np.nan, device=device), rtol)
    assert_close(derot_err, utils.rotate_hypercube(err, -posangs, cent=cent, cval0=np.nan, device=device), rtol)
    np.testing.assert_array_equal(derot[1], res[1]) # an angle of exactly zero leaves the image untouched
    np.testing.assert_array_equal(cent_out, cent)

    padded, _, cent_pad = rdi.rdi_residuals(hcube, hcube_ref, optzones, subzones, posangs=posangs, cent=cent, pad_before_derot=True, **kwargs)
    assert padded.shape[-2:] > res.shape[-2:] and np.all(cent_pad >= cent)

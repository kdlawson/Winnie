import hashlib
import numbers
import os
import shutil
import sys

import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from pyklip.klip import _rotate_wcs_hdr
from scipy import ndimage, optimize
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm

from .convolution import get_jwst_psf_grid_inds
from .space import SpaceConvolution, SpaceRDI, SpaceReduction
from .utils import ang_size_to_px_size, c_to_c_osamp, pad_or_crop_image, xy_polar_ang_displacement

# Band reference wavelengths (microns) from cgisim; sub-bands (e.g. 1A) map to their band.
RC_CWAVEL = {'1': 0.575, '2': 0.660, '3': 0.730, '4': 0.825}
RC_DIAMETER = 2.36*u.meter

# Default corgisim PSF grid (lambda/D, deg): r=0, then 0.25 steps to 4.5 and 0.5 steps to 11, every 15 deg.
RC_GRID_RVALS = (0., *np.round(np.arange(1.5, 4.51, 0.25), 2), *np.round(np.arange(5., 11.01, 0.5), 2))
RC_GRID_THVALS = tuple(np.arange(0., 360., 15.))
RC_GRID_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.winnie', 'rc_psfgrids')


def read_rc_header(h0, h1):
    """
    Read the metadata Winnie needs from the headers of a corgidrp L3/L4
    Roman Coronagraph (RC) product.

    Parameters
    ----------
    h0: astropy.io.fits.Header
        Primary header.

    h1: astropy.io.fits.Header
        Header of the SCI extension.

    Returns
    -------
    info: dict
        Keys: posang (deg; NORTHANG, else PA_APER, else ROLL), c_star and
        c_coron (0-based x, y pixels; None if STARLOCX/Y are absent, as in
        default L3 products), pxscale
        (arcsec/pixel), wavel (microns), filter, mask, lyot, date, visit,
        psfref and satspots.
    """
    if h1.get('DATALVL') not in ('L3', 'L4'):
        raise ValueError(f"DATALVL={h1.get('DATALVL')}; Winnie requires L3/L4 products from corgidrp.")
    # Default coronagraphic L4 (l3_to_l4 recipe) is already PSF-subtracted: a KL-mode cube, not input data.
    if h1['DATALVL'] == 'L4' and (h0.get('PSFSUB') or h1.get('PSFSUB')):
        raise ValueError(f"PSF-subtracted L4 (PSFSUB={h0.get('PSFSUB') or h1.get('PSFSUB')}); use the L3 frames.")
    # NORTHANG (corgidrp create_wcs) adds the astrometric north angle to the roll; corgidrp derotates by it.
    if 'NORTHANG' in h1:
        posang = h1['NORTHANG']
    elif 'PA_APER' in h0:
        posang = h0['PA_APER']
    elif 'ROLL' in h0:
        posang = h0['ROLL']
    else:
        raise ValueError('No NORTHANG, PA_APER or ROLL in headers.')
    band = h1['CFAMNAME'][0]
    if band not in RC_CWAVEL:
        raise ValueError(f"Unknown CFAM band: {h1['CFAMNAME']}")

    c_star = np.array([h1['STARLOCX'], h1['STARLOCY']], dtype=float) if 'STARLOCX' in h1 else None
    return {'posang': float(posang),
            'c_star': c_star,
            # No separate FPM center keyword yet; corgidrp treats STARLOC as the FPM center.
            'c_coron': None if c_star is None else c_star.copy(),
            'pxscale': h1['PLTSCALE'] / 1000.,
            'wavel': RC_CWAVEL[band],
            'filter': h1['CFAMNAME'],
            'mask': h1['FPAMNAME'],
            'lyot': h1['LSAMNAME'],
            'date': h1['FTIMEUTC'],
            'visit': h0['VISITID'],
            'psfref': bool(h0['PSFREF']),
            'satspots': bool(h1.get('SATSPOTS', 0))}


def find_rc_star(files, **find_star_kws):
    """
    Locate the star in RC L3 frames with corgidrp's find_star, using the
    satellite spot frames of the same visit. Input files are not modified.

    Parameters
    ----------
    files: list of str
        L3 files of a single visit: science frames (SATSPOTS=0) and the three
        groups of satellite spot frames (SATSPOTS=1).

    **find_star_kws:
        Passed to corgidrp.l3_to_l4.find_star.

    Returns
    -------
    c_stars: dict
        0-based (x, y) star position for each science file.
    """
    from corgidrp import data, l3_to_l4

    dataset = data.Dataset([os.path.abspath(f) for f in files])
    out = l3_to_l4.find_star(dataset, **find_star_kws)
    return {os.path.abspath(frame.filepath): np.array([frame.ext_hdr['STARLOCX'], frame.ext_hdr['STARLOCY']])
            for frame in out}


class RCDatabase:
    """
    Minimal stand-in for the spaceKLIP Database for Roman Coronagraph (RC)
    data. Holds only what SpaceRDI uses: obs (a dict of astropy Tables, one
    per concatenation), output_dir and red.
    """

    def __init__(self, output_dir):
        # SpaceRDI builds paths by string concatenation, so keep a trailing slash.
        self.output_dir = os.path.join(os.path.abspath(output_dir), '')
        os.makedirs(self.output_dir, exist_ok=True)
        self.obs = {}
        self.red = {}

    def add_concat(self, sci_files, ref_files, pxscale=None, cwavel=None, concat=None,
                   blurfwhm=0.0, maskfile='NONE'):
        """
        Add a concatenation of SCI and REF files to the database. Inputs are
        default corgidrp L3 products; satellite spot frames (SATSPOTS=1) may be
        included and are used only to locate the star (via find_rc_star) for
        visits whose headers lack STARLOCX/Y.

        Parameters
        ----------
        sci_files, ref_files: list of str
            L3 files of the science and reference visits.

        pxscale: float
            Pixel scale in arcsec/pixel. If not provided, read from the header
            of the first SCI file.

        cwavel: float
            Central wavelength in microns. If not provided, read from the
            header of the first SCI file.

        concat: str
            Name of the concatenation. Defaults to 'RC_<n>'.

        Returns
        -------
        concat: str
            Name of the added concatenation.
        """
        all_files = [os.path.abspath(f) for f in [*sci_files, *ref_files]]
        missing = [f for f in all_files if not os.path.isfile(f)]
        if missing:
            raise ValueError(f'Files not found: {missing}')
        if concat is None:
            concat = f'RC_{len(self.obs)}'
        if concat in self.obs:
            raise ValueError(f'Concatenation {concat} already exists.')

        infos = {f: read_rc_header(fits.getheader(f, 0), fits.getheader(f, 1)) for f in all_files}
        sci = [f for f in all_files[:len(sci_files)] if not infos[f]['satspots']]
        ref = [f for f in all_files[len(sci_files):] if not infos[f]['satspots']]
        if len(sci) == 0:
            raise ValueError('At least one SCI file (SATSPOTS=0) is required.')
        files = sci + ref

        c_stars = {f: infos[f]['c_star'] for f in files if infos[f]['c_star'] is not None}
        for visit in sorted({infos[f]['visit'] for f in files if f not in c_stars}):
            visit_files = [f for f in all_files if infos[f]['visit'] == visit]
            if not any(infos[f]['satspots'] for f in visit_files):
                raise ValueError(f'Visit {visit} has no STARLOCX/Y and no satellite spot frames to locate the star.')
            c_stars.update(find_rc_star(visit_files))

        info = infos[sci[0]]
        pxscale = info['pxscale'] if pxscale is None else pxscale
        cwavel = info['wavel'] if cwavel is None else cwavel

        n = len(files)
        tab = Table()
        tab['TYPE'] = ['SCI']*len(sci) + ['REF']*len(ref)
        tab['FITSFILE'] = np.array(files, dtype=object)
        tab['STARX'] = np.array([c_stars[f][0] for f in files], dtype=float)
        tab['STARY'] = np.array([c_stars[f][1] for f in files], dtype=float)
        tab['PIXSCALE'] = np.full(n, pxscale, dtype=float)
        tab['CWAVEL'] = np.full(n, cwavel, dtype=float)
        tab['BLURFWHM'] = np.full(n, blurfwhm, dtype=float)
        tab['EXP_TYPE'] = ['RC_CORON']*n
        tab['INSTRUME'] = ['ROMAN_CORONAGRAPH']*n
        tab['MASKFILE'] = np.array([maskfile]*n, dtype=object)
        self.obs[concat] = tab
        return concat

    def add_product(self, concat, filename):
        """
        Record a saved Winnie product in red[concat], replacing any entry with
        the same file (the RC stand-in for spaceKLIP's read_jwst_s3_data).
        """
        row = {'FITSFILE': os.path.abspath(filename),
               'OUTPUT_EXT': fits.getheader(filename, 0).get('OUTPUT_EXT', '')}
        if concat not in self.red:
            self.red[concat] = Table(names=list(row), dtype=[object, object])
        tab = self.red[concat]
        dup = np.where(tab['FITSFILE'] == row['FITSFILE'])[0]
        if len(dup):
            tab.remove_rows(dup)
        tab.add_row(row)


class RCReduction(SpaceReduction):
    """
    SpaceReduction for RC products. Derotated WCS headers are rotated by the
    roll from read_rc_header (NORTHANG), since RC headers have no ROLL_REF.
    """

    def __init__(self, file_to_load=None, spacerdi=None, output_ext=None, im=None, rolls=None,
                 derotated=True, **kwargs):
        new = spacerdi is not None and (im is not None or rolls is not None)
        if not new and file_to_load is not None and output_ext is None:
            # Avoids the parent's JWST filename parsing.
            output_ext = fits.getheader(file_to_load, 0)['OUTPUT_EXT']
        super().__init__(file_to_load=file_to_load, spacerdi=spacerdi, output_ext=output_ext, im=im,
                         rolls=rolls, derotated=(derotated and not new), **kwargs)
        if new and derotated:
            # Image headers come from the first frame of each visit; the combined header copies the first.
            uni_visit_inds = np.sort(np.unique(spacerdi._visit_ids_sci, return_index=True)[1])
            posangs = spacerdi._posangs_sci[uni_visit_inds]
            for h, posang in zip([*self.image_headers, self.combined_image_header], [*posangs, posangs[0]]):
                w = wcs.WCS(h, naxis=[1, 2])
                _rotate_wcs_hdr(w, posang)
                h['CD1_1'], h['CD1_2'] = w.wcs.cd[0]
                h['CD2_1'], h['CD2_2'] = w.wcs.cd[1]
            self.derotated = True
            self.primary_header['DEROT'] = True


class RCInstrument:
    """
    Instrument setup handed to RC PSF grid and transmission functions, in
    place of the stpsf/webbpsf_ext instruments used for JWST.
    """

    def __init__(self, pxscale, wavel, mask=None, lyot=None, filt=None):
        self.pxscale = pxscale << u.arcsec/u.pixel
        self.wavel = wavel << u.micron
        self.mask = mask
        self.lyot = lyot
        self.filt = filt

    @property
    def lamd(self):
        """lambda/D in arcsec."""
        return np.rad2deg((self.wavel/RC_DIAMETER.to(u.micron)).value)*3600


def gaussian_psf_grid(inst, source_spectrum=None, shift=None, osamp=2, fov_pixels=151, show_progress=True,
                      fwhm=None, rvals_lamd=(0, 2, 4, 6, 8, 10, 12), ntheta=8):
    """
    Synthetic PSF grid of identical Gaussians, for testing RC convolution
    before corgisim grids exist. Follows the grid_fn convention of
    SpaceConvolution.fetch_psf_grid.

    Parameters
    ----------
    inst: RCInstrument
    fwhm: float
        Gaussian FWHM in detector pixels. Default is lambda/D.
    rvals_lamd: sequence
        Sample separations from the mask center, in lambda/D; r=0 gets a
        single sample.
    ntheta: int
        Number of position angles per nonzero separation.

    Returns
    -------
    psfs, psfs, offsets_polar, offsets_xy
        PSFs (odd side, each summing to 1) twice (RC has no distortion, so
        "distorted" and "undistorted" are the same), offsets as [r (arcsec),
        theta (deg, CCW from +Y)] and as [x, y] (arcsec), each of shape (2, N).
    """
    if fwhm is None:
        fwhm = inst.lamd/inst.pxscale.value
    n = fov_pixels*osamp + (fov_pixels*osamp + 1) % 2
    y, x = np.indices((n, n)) - (n - 1)/2
    sig = fwhm*osamp/np.sqrt(8*np.log(2))
    psf = np.exp(-(x**2 + y**2)/(2*sig**2))
    psf = (psf/psf.sum()).astype(np.float32)
    rvals = np.asarray(rvals_lamd, dtype=float)*inst.lamd
    thetas = np.arange(ntheta)*360./ntheta
    r, theta = np.array([(rv, th) for rv in rvals for th in (thetas if rv > 0 else [0.])]).T
    offsets_polar = np.array([r, theta])
    offsets_xy = np.array([-r*np.sin(np.deg2rad(theta)), r*np.cos(np.deg2rad(theta))])
    psfs = np.repeat(psf[np.newaxis], len(r), axis=0)
    return psfs, psfs, offsets_polar, offsets_xy


def _rc_spectral_weights(optics, source_spectrum=None):
    """Photon weights per corgisim wavelength, as in CorgiOptics.construct_image_array (flat photlam by default)."""
    from synphot import Observation, SourceSpectrum, units
    from synphot.models import ConstFlux1D
    if source_spectrum is None:
        source_spectrum = SourceSpectrum(ConstFlux1D, amplitude=1*units.PHOTLAM)
    obs = Observation(source_spectrum, optics.bp)
    lam = np.asarray(optics.lam_um)
    dlam = lam[1] - lam[0] if len(lam) > 1 else 0.
    w = np.array([obs.countrate(area=optics.area, waverange=[(l - dlam/2)*1e4, (l + dlam/2)*1e4]).value for l in lam])
    return w/w.sum()


def corgisim_image(optics, dx_mas, dy_mas, nfine, sfine, weights, pool=None, **overrides):
    """
    Noiseless band-weighted corgisim intensity for a point source at EXCAM
    offset (dx_mas, dy_mas), on an (nfine, nfine) array at sfine samples per
    detector pixel, optical axis at pixel nfine//2. overrides update the
    roman_preflight keywords (e.g. use_fpm=0). pool: optional
    multiprocessing pool reused across calls (one wavelength per worker).
    """
    import proper
    import roman_preflight_proper
    # PROPER imports the prescription by name; spawned workers inherit sys.path.
    if roman_preflight_proper.lib_dir not in sys.path:
        sys.path.append(roman_preflight_proper.lib_dir)
    kw = dict(optics.optics_keywords, output_dim=int(nfine), final_sampling_m=optics.sampling_um*1e-6/sfine,
              source_x_offset_mas=float(dx_mas), source_y_offset_mas=float(dy_mas), **overrides)
    if pool is None:
        fields, _ = proper.prop_run_multi('roman_preflight', optics.lam_um, 1024, PASSVALUE=kw, QUIET=True)
    else:
        # Same worker call as prop_run_multi, without starting a new pool (spawned workers re-import the caller).
        params = [('roman_preflight', lam, 1024, kw, True, False) for lam in optics.lam_um]
        fields = np.stack([res[0] for res in pool.map(proper.prop_execute_multi, params)])
    return np.tensordot(weights, np.abs(fields)**2, axes=1)


def _register(ref, im):
    """Subpixel shift [dx, dy] such that im is ref shifted by it (Fourier shift, least squares)."""
    fref = np.fft.fft2(ref)
    xc = np.fft.ifft2(np.fft.fft2(im)*np.conj(fref)).real
    ny, nx = ref.shape
    iy, ix = np.unravel_index(np.argmax(xc), xc.shape)
    d0 = np.array([(ix + nx//2) % nx - nx//2, (iy + ny//2) % ny - ny//2], dtype=float)
    cost = lambda d: np.sum((np.fft.ifft2(ndimage.fourier_shift(fref, d[::-1])).real - im)**2)
    return optimize.minimize(cost, d0, method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-20)).x


def _hash(*items):
    h = hashlib.sha1()
    for item in items:
        if isinstance(item, dict):
            for k in sorted(item):
                h.update(k.encode())
                h.update(_hash(item[k]).encode())
        elif isinstance(item, np.ndarray):
            h.update(np.ascontiguousarray(item).tobytes())
        else:
            h.update(repr(item).encode())
    return h.hexdigest()[:12]


def _package_version(name):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return 'none'


def rc_psf_grid_key(optics, rvals_lamd=RC_GRID_RVALS, thvals_deg=RC_GRID_THVALS, osamp=2, fov_pixels=41,
                    source_spectrum=None, subsample=3, sample_fn=None):
    """Hash of every input that changes a corgisim PSF grid (used for its cache file name)."""
    skip = ('output_dim', 'final_sampling_m', 'source_x_offset_mas', 'source_y_offset_mas')
    okw = {k: (np.asarray(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
           for k, v in optics.optics_keywords.items() if k not in skip}
    spec = 'flat' if source_spectrum is None else np.asarray(source_spectrum(np.asarray(optics.lam_um)*1e4).value)
    fn = 'corgisim' if sample_fn is None else f'{sample_fn.__module__}.{sample_fn.__qualname__}'
    return _hash(optics.cgi_mode, optics.cor_type, optics.bandpass, np.asarray(optics.lam_um, dtype=float),
                 float(optics.res_mas), okw,
                 int(osamp), int(fov_pixels), int(subsample), np.asarray(rvals_lamd, dtype=float),
                 np.asarray(thvals_deg, dtype=float), spec, fn,
                 _package_version('corgisim'), _package_version('roman_preflight_proper'))


def rc_psf_grid_samples(optics, rvals_lamd=RC_GRID_RVALS, thvals_deg=RC_GRID_THVALS, osamp=2, fov_pixels=41,
                        source_spectrum=None, subsample=3, cache_dir=None, show_progress=True, sample_fn=None):
    """
    Noiseless corgisim PSF grid on a polar grid about the mask center, cached
    on disk. Each sample is normalized to its own total (Kellen's
    exit-pupil-like convention), so the mask and field-stop attenuation is
    returned separately as `throughput` for the transmission map.

    PSFs are centered on where the source actually lands. That position comes
    from a linear map (detector px = b + A @ offset_mas), measured once from
    unocculted PSFs, so it absorbs corgisim's effective plate scale and the
    tilt from its wavefront errors.

    corgisim runs PROPER with multiprocessing, so scripts calling this on a
    fresh grid need an `if __name__ == '__main__':` guard.

    Parameters
    ----------
    optics: corgisim.instrument.CorgiOptics
        Set up as for the data (mode, band, DM solution, errors).
    rvals_lamd, thvals_deg: sequences
        Grid radii (lambda/D, at optics.res_mas) and position angles (deg,
        counterclockwise from +Y); r=0 gets a single sample.
    osamp, fov_pixels: int
        Oversampling and PSF size in detector pixels; PSFs have an odd side
        of fov_pixels*osamp (+1 if even).
    subsample: int
        Odd number of propagation samples per oversampled pixel, binned for
        pixel integration.
    cache_dir: str
        Shared cache folder; default is ~/.winnie/rc_psfgrids. Samples are
        saved as they finish, so an interrupted run resumes.
    sample_fn: callable
        Replaces corgisim_image (same signature), e.g. for tests.

    Returns
    -------
    dict
        psfs (N, n, n) float32; rvals_lamd, thvals_deg (N); throughput (N,
        total relative to the unocculted on-axis source); offset_map_A (2, 2)
        in px/mas and offset_map_b (2) in px; norm (unocculted total); key.
    """
    if subsample % 2 == 0:
        raise ValueError('subsample must be odd, so binned pixels stay centered on the optical axis.')
    key = rc_psf_grid_key(optics, rvals_lamd, thvals_deg, osamp, fov_pixels, source_spectrum, subsample, sample_fn)
    cache_dir = RC_GRID_CACHE_DIR if cache_dir is None else cache_dir
    path = os.path.join(cache_dir, f'rc_psfgrid_{key}.fits')
    if os.path.isfile(path):
        return _read_rc_psf_grid(path)

    parts = os.path.join(cache_dir, f'rc_psfgrid_{key}_parts')
    os.makedirs(parts, exist_ok=True)
    if sample_fn is not None:
        return _rc_psf_grid_compute(optics, sample_fn, None, rvals_lamd, thvals_deg, osamp, fov_pixels, subsample,
                                    parts, path, key, show_progress)
    import multiprocessing
    import roman_preflight_proper
    if roman_preflight_proper.lib_dir not in sys.path:
        sys.path.append(roman_preflight_proper.lib_dir)
    weights = _rc_spectral_weights(optics, source_spectrum)
    with multiprocessing.Pool(min(len(optics.lam_um), multiprocessing.cpu_count())) as pool:
        sample = lambda *args, **kw: corgisim_image(*args, pool=pool, **kw)
        return _rc_psf_grid_compute(optics, sample, weights, rvals_lamd, thvals_deg, osamp, fov_pixels, subsample,
                                    parts, path, key, show_progress)


def _rc_psf_grid_compute(optics, sample_fn, weights, rvals_lamd, thvals_deg, osamp, fov_pixels, subsample, parts,
                         path, key, show_progress):
    k = int(osamp)*int(subsample)
    n = fov_pixels*osamp + (fov_pixels*osamp + 1) % 2

    def image(dx, dy, half, **overrides):
        """Propagate on (2*half*osamp+1)*subsample fine pixels, bin to osamp; axis at the central pixel."""
        m = 2*int(half)*osamp + 1
        im = sample_fn(optics, dx, dy, m*subsample, k, weights, **overrides)
        return im.reshape(m, subsample, m, subsample).sum(axis=(1, 3))

    calib_file = os.path.join(parts, 'calib.npz')
    if os.path.isfile(calib_file):
        calib = dict(np.load(calib_file))
    else:
        # Unocculted, no field stop: offset map from on-axis and +x/+y sources; b from the aberration-free on-axis PSF.
        o = 6*optics.res_mas
        unocc = dict(use_fpm=0, use_field_stop=0)
        on = image(0., 0., 30, **unocc)
        perfect = image(0., 0., 30, use_errors=0, use_dm1=0, use_dm2=0, **unocc)
        b = _register(perfect, on)/osamp
        A = np.array([_register(on, image(o, 0., 30, **unocc)), _register(on, image(0., o, 30, **unocc))]).T/(osamp*o)
        norm = image(0., 0., 50, use_fpm=0).sum()
        calib = dict(A=A, b=b, norm=norm)
        np.savez(calib_file, **calib)

    r_all, th_all = _rc_grid_positions(rvals_lamd, thvals_deg)
    iterator = tqdm(range(len(r_all)), leave=False) if show_progress else range(len(r_all))
    for i in iterator:
        part = os.path.join(parts, f'sample_{i:04d}.npz')
        if os.path.isfile(part):
            continue
        t = np.deg2rad(th_all[i])
        offset = r_all[i]*optics.res_mas*np.array([-np.sin(t), np.cos(t)])
        p = calib['b'] + calib['A'] @ offset
        im = image(*offset, max(50, int(np.ceil(np.hypot(*p) + fov_pixels/2 + 3))))
        c = (im.shape[0] - 1)/2
        psf = pad_or_crop_image(im, (n, n), cent=c + p*osamp, new_cent=((n - 1)/2, (n - 1)/2), cval0=0.)
        total = im.sum()
        # A fully occulted sample (zero flux) keeps a zero PSF; its transmission is zero anyway.
        np.savez(part, psf=(psf/total if total > 0 else psf).astype(np.float32), total=total)

    samples = [np.load(os.path.join(parts, f'sample_{i:04d}.npz')) for i in range(len(r_all))]
    grid = dict(psfs=np.array([s['psf'] for s in samples], dtype=np.float32), rvals_lamd=r_all, thvals_deg=th_all,
                throughput=np.array([float(s['total']) for s in samples])/calib['norm'],
                offset_map_A=calib['A'], offset_map_b=calib['b'], norm=float(calib['norm']), key=key)
    _write_rc_psf_grid(path, grid, optics, osamp, fov_pixels, subsample)
    shutil.rmtree(parts)
    return grid


def _rc_grid_positions(rvals_lamd, thvals_deg):
    r, th = np.array([(r, th) for r in rvals_lamd for th in (thvals_deg if r > 0 else [0.])], dtype=float).T
    return r, th


def _write_rc_psf_grid(path, grid, optics, osamp, fov_pixels, subsample):
    h = fits.Header()
    h['GRIDKEY'] = grid['key']
    h['CGIMODE'], h['CORTYPE'], h['BANDPASS'] = optics.cgi_mode, optics.cor_type, str(optics.bandpass)
    h['RES_MAS'] = (float(optics.res_mas), 'lambda/D used for the grid radii (mas)')
    h['OSAMP'], h['FOVPIX'], h['SUBSAMP'] = int(osamp), int(fov_pixels), int(subsample)
    h['NORM'] = (grid['norm'], 'unocculted on-axis total (FPM out)')
    hdus = [fits.PrimaryHDU(grid['psfs'], header=h),
            fits.ImageHDU(np.array([grid['rvals_lamd'], grid['thvals_deg']]), name='POSITIONS'),
            fits.ImageHDU(grid['throughput'], name='THROUGHPUT'),
            fits.ImageHDU(np.array(grid['offset_map_A']), name='OFFSET_A'),
            fits.ImageHDU(np.array(grid['offset_map_b']), name='OFFSET_B')]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _read_rc_psf_grid(path):
    with fits.open(path) as hdul:
        r, th = hdul['POSITIONS'].data
        return dict(psfs=hdul[0].data.astype(np.float32), rvals_lamd=r, thvals_deg=th,
                    throughput=hdul['THROUGHPUT'].data.astype(float), offset_map_A=hdul['OFFSET_A'].data,
                    offset_map_b=hdul['OFFSET_B'].data, norm=hdul[0].header['NORM'], key=hdul[0].header['GRIDKEY'],
                    res_mas=hdul[0].header['RES_MAS'])


def rc_grid_offsets(grid, res_mas, pxscale):
    """
    Grid sample offsets from the mask center for Winnie, as ([r (arcsec),
    theta (deg)], [x, y] (arcsec)) in the data frame: the measured offset map
    applied as an isotropic scale and rotation (keeping the polar grid
    regular), converted with the data pixel scale.
    """
    A = np.asarray(grid['offset_map_A'])
    scale = np.sqrt(abs(np.linalg.det(A)))
    rot = np.rad2deg(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))
    r = grid['rvals_lamd']*res_mas*scale*u.Quantity(pxscale, u.arcsec/u.pixel).value
    th = np.where(grid['rvals_lamd'] > 0, np.mod(grid['thvals_deg'] + rot, 360.), grid['thvals_deg'])
    xy = np.array([-r*np.sin(np.deg2rad(th)), r*np.cos(np.deg2rad(th))])
    return np.array([r, th]), xy


def generate_rc_psf_grid(inst, source_spectrum=None, shift=None, osamp=2, fov_pixels=41, show_progress=True,
                         optics=None, rvals_lamd=RC_GRID_RVALS, thvals_deg=RC_GRID_THVALS, subsample=3,
                         cache_dir=None, sample_fn=None):
    """
    corgisim PSF grid for SpaceConvolution.fetch_psf_grid (grid_fn
    convention). See rc_psf_grid_samples; `optics` (a CorgiOptics set up as
    for the data) is required. Pair with rc_transmission_map, which
    RCConvolution does by default.

    Returns
    -------
    psfs, psfs, offsets_polar, offsets_xy
    """
    if optics is None:
        raise ValueError('generate_rc_psf_grid needs grid_kwargs=dict(optics=CorgiOptics(...)).')
    grid = rc_psf_grid_samples(optics, rvals_lamd, thvals_deg, osamp=osamp, fov_pixels=fov_pixels,
                               source_spectrum=source_spectrum, subsample=subsample, cache_dir=cache_dir,
                               show_progress=show_progress, sample_fn=sample_fn)
    offsets_polar, offsets_xy = rc_grid_offsets(grid, optics.res_mas, inst.pxscale)
    return grid['psfs'], grid['psfs'], offsets_polar, offsets_xy


def rc_transmission_map(inst, c_coron, osamp=2, shape=None, return_oversample=True, posang=0, c_star=None,
                        nd_squares=None, optics=None, fov_pixels=41, source_spectrum=None, **grid_kwargs):
    """
    Coronagraph transmission map (FPM and field stop) from the throughput of
    the cached corgisim PSF grid, interpolated linearly in (r, theta) on the
    polar grid (theta periodic; r clamped to the outermost ring). Same
    convention as convolution.get_jwst_coron_transmission_map; nd_squares is
    ignored.
    """
    ny, nx = shape
    grid = rc_psf_grid_samples(optics, osamp=osamp, fov_pixels=fov_pixels, source_spectrum=source_spectrum,
                               **grid_kwargs)
    polar, _ = rc_grid_offsets(grid, optics.res_mas, inst.pxscale)
    yg, xg = c_to_c_osamp(np.indices((ny*osamp, nx*osamp), dtype=np.float64), 1/osamp)
    x, y = xg - c_coron[0], yg - c_coron[1]
    if posang != 0:
        c_star = c_coron if c_star is None else c_star
        x, y = xy_polar_ang_displacement(x + c_coron[0] - c_star[0], y + c_coron[1] - c_star[1], -posang)
        x, y = x - c_coron[0] + c_star[0], y - c_coron[1] + c_star[1]
    # Pixel (r, theta) in grid units: lambda/D and the grid's own PAs (rc_grid_offsets applies the offset map).
    px_per_lamd = polar[0][grid['rvals_lamd'] > 0][0]/grid['rvals_lamd'][grid['rvals_lamd'] > 0][0]/inst.pxscale.value
    rot = (polar[1] - grid['thvals_deg'])[grid['rvals_lamd'] > 0][0]
    r = np.hypot(x, y)/px_per_lamd
    th = np.mod(np.rad2deg(np.arctan2(-x, y)) - rot, 360.)
    tmap = _polar_interp(grid['rvals_lamd'], grid['thvals_deg'], grid['throughput'], r, th)
    if return_oversample:
        return tmap
    return tmap.reshape(ny, osamp, nx, osamp).mean(axis=(1, 3))


def _polar_interp(r_s, th_s, vals, r, th):
    """Bilinear interpolation of samples on a regular polar grid (one r=0 sample allowed) at (r, th)."""
    rings = np.unique(r_s[r_s > 0])
    ths = np.unique(th_s[r_s > 0])
    table = np.full((len(rings), len(ths)), np.nan)
    for v, rv, tv in zip(vals, r_s, th_s):
        if rv > 0:
            table[np.searchsorted(rings, rv), np.searchsorted(ths, tv)] = v
    if np.any(~np.isfinite(table)):
        raise ValueError('rc_transmission_map needs a regular polar grid (every PA at every radius).')
    rr = np.concatenate([[0.], rings]) if np.any(r_s == 0) else rings
    table = np.concatenate([np.full((1, len(ths)), vals[r_s == 0][0]), table]) if np.any(r_s == 0) else table
    # Periodic in theta: append the first PA at +360.
    tt = np.concatenate([ths, [ths[0] + 360.]])
    table = np.concatenate([table, table[:, :1]], axis=1)
    th = np.where(th < ths[0], th + 360., th)
    interp = RegularGridInterpolator((rr, tt), table, method='linear')
    pts = np.stack([np.clip(r, rr[0], rr[-1]), th], axis=-1)
    return interp(pts.reshape(-1, 2)).reshape(np.shape(r))


def _grid_tag(grid_fn, grid_kwargs, fov_pixels, osamp, source_spectrum):
    """Short hash of the grid inputs, so the per-reduction grid file isn't reused for a different grid."""
    if grid_fn is generate_rc_psf_grid:
        kw = {k: v for k, v in grid_kwargs.items() if k in ('rvals_lamd', 'thvals_deg', 'subsample', 'sample_fn')}
        return rc_psf_grid_key(grid_kwargs['optics'], osamp=osamp, fov_pixels=fov_pixels,
                               source_spectrum=source_spectrum, **kw)[:8]
    return _hash(getattr(grid_fn, '__module__', ''), getattr(grid_fn, '__qualname__', repr(grid_fn)),
                 grid_kwargs, source_spectrum)[:8]


class RCConvolution(SpaceConvolution):
    """
    SpaceConvolution for RC data. The image geometry (rolls, star and mask
    centers, shape) comes from the RCRDI instance, since RC images are aligned,
    combined per visit and padded on load. The PSF grid and transmission
    functions receive an RCInstrument instead of stpsf/webbpsf_ext objects.
    """

    @property
    def inst_stpsf(self):
        return self._rc_inst

    @property
    def inst_webbpsfext(self):
        return self._rc_inst

    def calc_psf_shift(self, *args, **kwargs):
        self._psf_shift = None

    def __getstate__(self):
        # Always pickle the grid: refetching on unpickle would lose a custom grid.
        efficient_saving, self.efficient_saving = self.efficient_saving, False
        state = super().__getstate__()
        self.efficient_saving = efficient_saving
        return state

    def load_concat(self, concat, posangs, c_coron, c_star, shape, pxscale, wavel, blurfwhm=0., mask=None,
                    lyot=None, filt=None, fov_pixels=151, osamp=2, output_ext='psfs', prefetch_psf_grid=True,
                    recalc_psf_grid=False, cropped_shape=None, grid_fn=None, grid_kwargs={}, grid_inds_fn=None,
                    grid_inds_kwargs={}, transmission_map_fn=None, transmission_map_kwargs={}, **kwargs):
        """
        Set up convolution for the SCI images of an RCRDI concatenation.

        Parameters
        ----------
        posangs, c_coron: arrays
            Roll (deg) and mask center ([x, y]) per SCI image.
        c_star, shape: arrays
            Star position ([x, y]) and [ny, nx] of the (padded, uncropped) images.
        grid_fn: callable
            PSF grid function, called as grid_fn(RCInstrument, source_spectrum=,
            shift=, osamp=, fov_pixels=, show_progress=, **grid_kwargs). Default
            is gaussian_psf_grid; use generate_rc_psf_grid with
            grid_kwargs=dict(optics=CorgiOptics(...)) for corgisim PSFs.
        transmission_map_fn: callable
            Called as transmission_map_fn(RCInstrument, c_coron, osamp=, shape=,
            **transmission_map_kwargs). Default (None) is rc_transmission_map
            for generate_rc_psf_grid and a flat map of ones otherwise.

        Other keywords (e.g. JWST-only reference_index, coron_offsets) are ignored.
        """
        if isinstance(concat, numbers.Number):
            concat = list(self.database.obs.keys())[concat]
        self.concat = concat
        self.osamp = int(osamp)
        self.fov_pixels = int(fov_pixels)
        self.pxscale = pxscale << u.arcsec/u.pixel
        self.blurfwhm = blurfwhm
        self.blursigma = blurfwhm/np.sqrt(8.*np.log(2.))
        self._rc_inst = RCInstrument(self.pxscale, wavel, mask=mask, lyot=lyot, filt=filt)
        self.posangs_sci = np.asarray(posangs)
        self._c_coron_sci = np.array(c_coron, dtype=float)
        self._c_coron_ref = np.empty((0, 2))
        self._c_star = np.array(c_star, dtype=float)
        self._ny, self._nx = shape

        self.grid_fn = gaussian_psf_grid if grid_fn is None else grid_fn
        self.grid_kwargs = grid_kwargs
        self.grid_inds_fn = get_jwst_psf_grid_inds if grid_inds_fn is None else grid_inds_fn
        self.grid_inds_kwargs = grid_inds_kwargs
        if transmission_map_fn is None and self.grid_fn is generate_rc_psf_grid:
            # Unit-total corgisim PSFs need the matching attenuation map.
            transmission_map_fn = rc_transmission_map
            transmission_map_kwargs = {**grid_kwargs, 'fov_pixels': self.fov_pixels,
                                       'source_spectrum': self.source_spectrum, 'show_progress': False,
                                       **transmission_map_kwargs}
        self.transmission_map_fn = transmission_map_fn
        self.transmission_map_kwargs = transmission_map_kwargs

        tag = _grid_tag(self.grid_fn, grid_kwargs, self.fov_pixels, self.osamp, self.source_spectrum)
        base = f'{self.output_dir}{concat}_fov{self.fov_pixels}_os{self.osamp}_{tag}_{output_ext}'
        self._psf_file_dist, self._psf_file_nodist = f'{base}.fits', f'{base}_nodist.fits'
        self.psf_file = self._psf_file_nodist
        self._psf_shift = None
        self._coron_tmaps_osamp = None
        self._psf_inds_osamp = None

        self.cropped_shape = cropped_shape
        if prefetch_psf_grid:
            self.fetch_psf_grid(recalc_psf_grid=recalc_psf_grid)
        else:
            self._grid_fetched = False


class RCRDI(SpaceRDI):
    """
    SpaceRDI for Roman Coronagraph (RC) data, built from an RCDatabase of
    corgidrp L3 products. Distortion correction and coronagraph transmission
    maps are off by default, since they rely on JWST-only tools (pysiaf,
    webbpsf_ext).
    """
    reduction_class = RCReduction
    convolution_class = RCConvolution

    def __init__(self, database=None, data_ext=None, correct_distortion=False, distort_models=False,
                 save_coron_transmission=False, **kwargs):
        if data_ext is None and kwargs.get('from_fits') is None:
            # corgidrp names end in '_l3_.fits', so take the last non-empty token.
            f = database.obs[list(database.obs.keys())[0]]['FITSFILE'][0]
            data_ext = [t for t in os.path.basename(f).removesuffix('.fits').split('_') if t][-1]
        super().__init__(database=database, data_ext=data_ext, correct_distortion=correct_distortion,
                         distort_models=distort_models, save_coron_transmission=save_coron_transmission,
                         **kwargs)

    def __getstate__(self):
        # Always pickle the data: re-reading files on unpickle would skip alignment and DQ masking.
        efficient_saving, self.efficient_saving = self.efficient_saving, False
        state = super().__getstate__()
        self.efficient_saving = efficient_saving
        return state

    def _load_convolver(self, **kwargs):
        kwargs = {**kwargs, 'cropped_shape': self.cropped_shape}
        self.convolver.load_concat(self.concat, posangs=self._posangs_sci, c_coron=self._c_coron_sci,
                                   c_star=self._c_star, shape=(self._ny, self._nx), pxscale=self.pxscale,
                                   wavel=self._lam, blurfwhm=self._blurfwhm, mask=self._image_mask,
                                   lyot=self._pupil_mask, filt=self.filt, **kwargs, **self.convolver_args)

    def _save_products(self, products, include_instance=False, add_output_to_db=False):
        super()._save_products(products, include_instance=include_instance, add_output_to_db=False)
        if add_output_to_db:
            self.database.add_product(self.concat, products.filename)

    def load_concat(self, concat, cropped_shape=None, align=True, combine_frames=True):
        """
        Load data for an indicated concatenation in preparation for an RDI
        reduction.

        Parameters
        ----------
        concat: str or int
            Either the full concatenation string or the index of the desired
            concatenation.

        cropped_shape: array or tuple
            Spatial dimensions to which data should be cropped, as [ny,nx].

        align: bool
            If True, shifts every SCI and REF frame (and its error) so that its
            star (STARX/STARY in the database) lands on the star of the first
            SCI frame, as spaceKLIP's align_frames does for JWST data. If
            False, frames are used as-is and the star of the first SCI frame is
            assumed for all of them.

        combine_frames: bool
            If True, combines the (aligned) frames of each visit with
            self.input_combine_fn, giving one image per roll or reference
            visit, as SpaceRDI does with the integrations of a JWST file. RC
            visits are many short L3 frames, one per file. If False, every
            frame is kept as its own image.
        """
        if isinstance(concat, numbers.Number):
            concat_str = list(self.database.obs.keys())[concat]
        else:
            concat_str = concat

        db_tab = self.database.obs[concat_str]
        sci = np.asarray(db_tab['TYPE'] == 'SCI')
        ref = np.asarray(db_tab['TYPE'] == 'REF')
        files = np.array(db_tab['FITSFILE'])
        c_stars = np.array([db_tab['STARX'], db_tab['STARY']], dtype=float).T

        self.pxscale = db_tab[0]['PIXSCALE']*u.arcsec/u.pixel
        self._lam = db_tab['CWAVEL'][0]*u.micron
        self._blurfwhm = np.nan_to_num(db_tab['BLURFWHM'][0])
        self._d_eff = RC_DIAMETER
        self._instfwhm = ang_size_to_px_size(np.rad2deg((self._lam/(self._d_eff.to(u.micron))).value)*u.deg, self.pxscale).value
        self._fwhm = np.hypot(self._instfwhm, self._blurfwhm)
        self._sigma = self._fwhm/np.sqrt(8.*np.log(2.))
        self.concat = concat_str
        self._prepared_refmats = None

        self._c_star = c_stars[np.argmax(sci)].copy()
        if not align and np.max(np.hypot(*(c_stars - self._c_star).T)) > 0.1:
            print('WARNING: star centers differ by more than 0.1 px between frames and align=False.')

        imcube, errcube, posangs, visit_ids, dates, c_coron = [], [], [], [], [], []
        for f, c_star_i in zip(files, c_stars):
            with fits.open(f) as hdul:
                ints = hdul[1].data.astype(float)
                errs = hdul['ERR'].data.astype(float) if self.prop_err else None
                dq = hdul['DQ'].data
                info = read_rc_header(hdul[0].header, hdul[1].header)
            if np.ndim(ints.squeeze()) != 2:
                im, err = self.input_combine_fn(ints, errs, **self.input_combine_kws)
            else:
                im, err = ints.squeeze(), (None if not self.prop_err else errs.squeeze())
            # Flagged pixels become NaN, as corgidrp's replace_bad_pixels does before interpolating.
            bad = dq.squeeze() if np.ndim(dq.squeeze()) == 2 else np.any(dq, axis=0)
            im = np.where(bad > 0, np.nan, im)
            err = None if err is None else np.where(bad > 0, np.nan, err)
            if align:
                im = pad_or_crop_image(im, im.shape, cent=c_star_i, new_cent=self._c_star)
                err = None if err is None else pad_or_crop_image(err, err.shape, cent=c_star_i, new_cent=self._c_star)
                c_coron.append(self._c_star.copy())
            else:
                # No FPM-center keyword in L3; corgidrp treats the star as the FPM center.
                c_coron.append(c_star_i)
            imcube.append(im)
            errcube.append(err)
            posangs.append(info['posang'])
            visit_ids.append(info['visit'])
            dates.append(info['date'])

        c_coron = np.array(c_coron)
        visit_ids, dates = np.array(visit_ids), np.array(dates)
        posangs = np.array(posangs)
        # Visits are grouped by VISITID, but corgidrp's mocks give every frame the same one: split such a VISITID
        # by type and roll so that different rolls (or SCI and REF) are never combined.
        keys = np.array([f"{v}_{'SCI' if s else 'REF'}{p:.2f}" for v, s, p in zip(visit_ids, sci, np.mod(posangs, 360))])
        for v in np.unique(visit_ids):
            if len(np.unique(keys[visit_ids == v])) > 1:
                visit_ids = np.where(visit_ids == v, keys, visit_ids)
        if combine_frames:
            # One image per visit (keeping its first file, date and roll), SCI visits first as in the table.
            first = np.sort(np.unique(visit_ids, return_index=True)[1])
            first = first[np.argsort(~sci[first], kind='stable')]
            combined = [self.input_combine_fn(np.array(imcube)[visit_ids == visit_ids[i]],
                                              np.array(errcube)[visit_ids == visit_ids[i]] if self.prop_err else None,
                                              **self.input_combine_kws) for i in first]
            imcube, errcube = [im for im, _ in combined], [err for _, err in combined]
            files, posangs, visit_ids, dates, c_coron = files[first], posangs[first], visit_ids[first], dates[first], c_coron[first]
            sci, ref = sci[first], ref[first]
        self._coron_offsets = self._c_star - c_coron
        self._image_mask = info['mask']
        self._pupil_mask = info['lyot']
        self._aperturename = None
        self._instrument = db_tab[0]['INSTRUME']
        self.filt = info['filter']

        self._imcube_sci = np.array(imcube)[sci]
        self._errcube_sci = np.array(errcube)[sci] if self.prop_err else None
        self._posangs_sci = np.array(posangs)[sci]
        self._visit_ids_sci = np.array(visit_ids)[sci]
        self._c_coron_sci = c_coron[sci]
        self._dates_sci = np.array(dates)[sci]
        self._files_sci = files[sci]

        self._imcube_ref = np.array(imcube)[ref]
        self._errcube_ref = np.array(errcube)[ref] if self.prop_err else None
        self._posangs_ref = np.array(posangs)[ref]
        self._visit_ids_ref = np.array(visit_ids)[ref]
        self._c_coron_ref = c_coron[ref]
        self._dates_ref = np.array(dates)[ref]
        self._files_ref = files[ref]

        self._ny, self._nx = self._imcube_sci.shape[-2:]

        if self.pad_data is not None: self._apply_padding()

        self._imcube_css = None
        self.cropped_shape = cropped_shape
        self.rdi_settings = {}
        self.fixed_rdi_settings = {}
        self.update_annular_zones(exclude_opt_nans=True)
        self.set_crop(cropped_shape)
        self.rdi_presets()
        self.siaf_aper = None

        if self.convolver is not None:
            self._load_convolver()

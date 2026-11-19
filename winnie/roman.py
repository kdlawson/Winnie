import numbers
import os

import astropy.units as u
import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from pyklip.klip import _rotate_wcs_hdr

from .space import SpaceRDI, SpaceReduction
from .utils import ang_size_to_px_size, pad_or_crop_image

# Band reference wavelengths (microns) from cgisim; sub-bands (e.g. 1A) map to their band.
RC_CWAVEL = {'1': 0.575, '2': 0.660, '3': 0.730, '4': 0.825}
RC_DIAMETER = 2.36*u.meter


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


class RCRDI(SpaceRDI):
    """
    SpaceRDI for Roman Coronagraph (RC) data, built from an RCDatabase of
    corgidrp L3 products. Distortion correction and coronagraph transmission
    maps are off by default, since they rely on JWST-only tools (pysiaf,
    webbpsf_ext).
    """
    reduction_class = RCReduction

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
            self.convolver.load_concat(self.concat, **self.convolver_args)

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table

# Band reference wavelengths (microns) from cgisim; sub-bands (e.g. 1A) map to their band.
RC_CWAVEL = {'1': 0.575, '2': 0.660, '3': 0.730, '4': 0.825}


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
        Keys: posang (deg), c_star and c_coron (0-based x, y pixels), pxscale
        (arcsec/pixel), wavel (microns), filter, mask, lyot, date, visit and
        psfref.
    """
    if 'STARLOCX' not in h1:
        raise ValueError(f"No STARLOCX in header (DATALVL={h1.get('DATALVL')}); "
                         "Winnie requires L3/L4 products from corgidrp.")
    if 'PA_APER' in h0:
        posang = h0['PA_APER']
    elif 'ROLL' in h0:
        posang = h0['ROLL']
    else:
        raise ValueError('No PA_APER or ROLL in primary header.')
    band = h1['CFAMNAME'][0]
    if band not in RC_CWAVEL:
        raise ValueError(f"Unknown CFAM band: {h1['CFAMNAME']}")

    c_star = np.array([h1['STARLOCX'], h1['STARLOCY']], dtype=float)
    return {'posang': float(posang),
            'c_star': c_star,
            # No separate FPM center keyword yet; corgidrp treats STARLOC as the FPM center.
            'c_coron': c_star.copy(),
            'pxscale': h1['PLTSCALE'] / 1000.,
            'wavel': RC_CWAVEL[band],
            'filter': h1['CFAMNAME'],
            'mask': h1['FPAMNAME'],
            'lyot': h1['LSAMNAME'],
            'date': h1['FTIMEUTC'],
            'visit': h0['VISITID'],
            'psfref': bool(h0['PSFREF'])}


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
        Add a concatenation of SCI and REF files to the database.

        Parameters
        ----------
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
        if len(sci_files) == 0:
            raise ValueError('At least one SCI file is required.')
        files = [os.path.abspath(f) for f in [*sci_files, *ref_files]]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            raise ValueError(f'Files not found: {missing}')
        if concat is None:
            concat = f'RC_{len(self.obs)}'
        if concat in self.obs:
            raise ValueError(f'Concatenation {concat} already exists.')
        if pxscale is None or cwavel is None:
            info = read_rc_header(fits.getheader(files[0], 0), fits.getheader(files[0], 1))
            pxscale = info['pxscale'] if pxscale is None else pxscale
            cwavel = info['wavel'] if cwavel is None else cwavel

        n = len(files)
        tab = Table()
        tab['TYPE'] = ['SCI']*len(sci_files) + ['REF']*len(ref_files)
        tab['FITSFILE'] = np.array(files, dtype=object)
        tab['PIXSCALE'] = np.full(n, pxscale, dtype=float)
        tab['CWAVEL'] = np.full(n, cwavel, dtype=float)
        tab['BLURFWHM'] = np.full(n, blurfwhm, dtype=float)
        tab['EXP_TYPE'] = ['RC_CORON']*n
        tab['INSTRUME'] = ['ROMAN_CORONAGRAPH']*n
        tab['MASKFILE'] = np.array([maskfile]*n, dtype=object)
        self.obs[concat] = tab
        return concat

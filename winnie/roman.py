import os

import numpy as np
from astropy.table import Table


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

    def add_concat(self, sci_files, ref_files, pxscale, cwavel, concat=None,
                   blurfwhm=0.0, maskfile='NONE'):
        """
        Add a concatenation of SCI and REF files to the database.

        Parameters
        ----------
        pxscale: float
            Pixel scale in arcsec/pixel.

        cwavel: float
            Central wavelength in microns.

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

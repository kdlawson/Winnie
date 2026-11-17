import os

import numpy as np
import pytest
from astropy.io import fits

# winnie imports webbpsf_ext/stpsf at module level, which fail without these set.
_data = os.path.expanduser('~/data')
os.environ.setdefault('STPSF_PATH', os.path.join(_data, 'stpsf-data'))
os.environ.setdefault('WEBBPSF_EXT_PATH', os.path.join(_data, 'webbpsf_ext_data'))
os.environ.setdefault('PYSYN_CDBS', os.path.join(os.environ['WEBBPSF_EXT_PATH'], 'cdbs'))

# Synthetic RC observation; header layout follows corgidrp L4 products.
NY, NX = 61, 61
PXSCALE = 0.0218  # arcsec/pixel
STAR_XY = (30.0, 30.0)  # 0-based
PLANET_SEP_PX = 12.0
PLANET_PA_DEG = 60.0  # east of north
PLANET_CONTRAST = 0.05
SCI_ROLLS = (-13.0, 13.0)
NREF = 2
REF_SCALE = 0.9  # REF star flux relative to SCI
NINTS = 3


def _gauss(x0, y0, fwhm=2.5, amp=1.0):
    y, x = np.mgrid[:NY, :NX]
    sig = fwhm / 2.3548
    return amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sig**2))


def planet_xy(roll):
    """Detector position of the planet for a given roll (PA_APER, deg)."""
    theta = np.deg2rad(PLANET_PA_DEG - roll)
    return STAR_XY[0] - PLANET_SEP_PX*np.sin(theta), STAR_XY[1] + PLANET_SEP_PX*np.cos(theta)


def _write_rc_file(path, roll, visitid, rng, psfref=False, planet=False):
    star = _gauss(*STAR_XY, fwhm=3.0) * (REF_SCALE if psfref else 1.0)
    img = star + (_gauss(*planet_xy(roll), amp=PLANET_CONTRAST) if planet else 0)
    sci = np.array([img + rng.normal(0, 1e-4, img.shape) for _ in range(NINTS)], dtype=np.float32)
    err = np.full_like(sci, 1e-4)

    h0 = fits.Header()
    h0['TELESCOP'] = 'ROMAN'
    h0['INSTRUME'] = 'CGI'
    h0['ORIGIN'] = 'CorgiSim'
    h0['VISITID'] = visitid
    h0['PSFREF'] = int(psfref)
    h0['PA_APER'] = roll
    h0['TARGET'] = 'REF' if psfref else 'SCI'

    h1 = fits.Header()
    h1['EXTNAME'] = 'SCI'
    h1['DATALVL'] = 'L4'
    h1['BUNIT'] = 'Photoelectrons / s'
    h1['STARLOCX'] = STAR_XY[0]
    h1['STARLOCY'] = STAR_XY[1]
    h1['PLTSCALE'] = PXSCALE * 1e3  # mas/pixel
    h1['CFAMNAME'] = '1F'
    h1['FPAMNAME'] = 'HLC12_C2R1'
    h1['LSAMNAME'] = 'NFOV'
    h1['FTIMEUTC'] = '2027-01-01T00:00:00'

    # Same extension layout as corgidrp.data.Image.save: SCI, ERR, DQ.
    fits.HDUList([fits.PrimaryHDU(header=h0),
                  fits.ImageHDU(sci, header=h1),
                  fits.ImageHDU(err, name='ERR'),
                  fits.ImageHDU(np.zeros(sci.shape, dtype=np.uint8), name='DQ')]).writeto(path, overwrite=True)


@pytest.fixture(scope='session')
def rc_fixture(tmp_path_factory):
    """Writes synthetic SCI (two rolls, with planet) and REF files; returns their paths."""
    d = tmp_path_factory.mktemp('rc_data')
    rng = np.random.default_rng(0)
    sci, ref = [], []
    for i, roll in enumerate(SCI_ROLLS):
        p = str(d / f'rc_sci_roll{i}.fits')
        _write_rc_file(p, roll, f'000010010010010010{i}', rng, psfref=False, planet=True)
        sci.append(p)
    for i in range(NREF):
        p = str(d / f'rc_ref_{i}.fits')
        _write_rc_file(p, 0.0, f'000010010010020010{i}', rng, psfref=True, planet=False)
        ref.append(p)
    return {'sci': sci, 'ref': ref, 'dir': str(d)}

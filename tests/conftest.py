import os

import numpy as np
import pytest
from astropy.io import fits

# winnie imports webbpsf_ext/stpsf at module level, which fail without these set.
_data = os.path.expanduser('~/data')
os.environ.setdefault('STPSF_PATH', os.path.join(_data, 'stpsf-data'))
os.environ.setdefault('WEBBPSF_EXT_PATH', os.path.join(_data, 'webbpsf_ext_data'))
os.environ.setdefault('PYSYN_CDBS', os.path.join(os.environ['WEBBPSF_EXT_PATH'], 'cdbs'))

# Synthetic RC observation; header layout follows default corgidrp L3 products.
NY, NX = 61, 61
PXSCALE = 0.0218  # arcsec/pixel
STAR_XY = (30.0, 30.0)  # 0-based
PLANET_SEP_PX = 12.0
PLANET_PA_DEG = 60.0  # east of north
PLANET_CONTRAST = 0.05
SCI_ROLLS = (-13.0, 13.0)
NREF = 2
REF_SCALE = 0.9  # REF star flux relative to SCI
# Per-frame star offsets (px) from STAR_XY, as find_star reports on unregistered L3 frames.
SCI_OFFSETS = ((0.0, 0.0), (0.3, -0.2))
REF_OFFSETS = ((-0.25, 0.15), (0.1, 0.35))
BAD_PIXEL = (45, 10)  # (y, x) flagged in DQ of the second REF file


def _gauss(x0, y0, fwhm=2.5, amp=1.0):
    y, x = np.mgrid[:NY, :NX]
    sig = fwhm / 2.3548
    return amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sig**2))


def planet_xy(roll, star_xy=STAR_XY):
    """Detector position of the planet for a given roll (PA_APER, deg)."""
    theta = np.deg2rad(PLANET_PA_DEG - roll)
    return star_xy[0] - PLANET_SEP_PX*np.sin(theta), star_xy[1] + PLANET_SEP_PX*np.cos(theta)


def _write_rc_file(path, roll, visitid, rng, psfref=False, planet=False, starloc=True,
                   star_xy=STAR_XY, bad_pixel=None):
    star = _gauss(*star_xy, fwhm=3.0) * (REF_SCALE if psfref else 1.0)
    img = star + (_gauss(*planet_xy(roll, star_xy), amp=PLANET_CONTRAST) if planet else 0)
    sci = (img + rng.normal(0, 1e-4, img.shape)).astype(np.float32)
    err = np.full((1, NY, NX), 1e-4, dtype=np.float32)
    dq = np.zeros((NY, NX), dtype=np.uint16)
    if bad_pixel is not None:
        sci[bad_pixel] = 1e3
        dq[bad_pixel] = 1

    h0 = fits.Header()
    h0['TELESCOP'] = 'ROMAN'
    h0['INSTRUME'] = 'CGI'
    h0['ORIGIN'] = 'CorgiSim'
    h0['VISITID'] = visitid
    h0['PSFREF'] = int(psfref)
    h0['PA_APER'] = roll % 360  # corgisim wraps to 0-360
    h0['TARGET'] = 'REF' if psfref else 'SCI'

    h1 = fits.Header()
    h1['EXTNAME'] = 'SCI'
    h1['DATALVL'] = 'L3'
    h1['BUNIT'] = 'photoelectron/s'
    h1['SATSPOTS'] = 0
    h1['FSMPRFL'] = 'NFOV'
    h1['EACQ_COL'] = STAR_XY[0]
    h1['EACQ_ROW'] = STAR_XY[1]
    # Default L3 has no STARLOCX/Y; starloc=True mimics a user-run corgidrp find_star.
    if starloc:
        h1['STARLOCX'] = star_xy[0]
        h1['STARLOCY'] = star_xy[1]
    h1['PLTSCALE'] = PXSCALE * 1e3  # mas/pixel
    h1['CFAMNAME'] = '1F'
    h1['FPAMNAME'] = 'HLC12_C2R1'
    h1['LSAMNAME'] = 'NFOV'
    h1['FTIMEUTC'] = '2027-01-01T00:00:00'
    # WCS as corgidrp l2b_to_l3.create_wcs writes it, with astrometric north angle 0.
    h1['NORTHANG'] = roll % 360
    ang = np.deg2rad(roll)
    cd = np.array([[-np.cos(ang), np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * PXSCALE / 3600
    h1['CTYPE1'], h1['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    h1['CRPIX1'], h1['CRPIX2'] = (NX - 1) // 2, (NY - 1) // 2
    h1['CRVAL1'], h1['CRVAL2'] = 0.0, 0.0
    h1['CD1_1'], h1['CD1_2'], h1['CD2_1'], h1['CD2_2'] = cd.ravel()

    # Same extension layout and shapes as corgidrp L3: SCI (ny, nx), ERR (1, ny, nx), DQ (ny, nx).
    fits.HDUList([fits.PrimaryHDU(header=h0),
                  fits.ImageHDU(sci, header=h1),
                  fits.ImageHDU(err, name='ERR'),
                  fits.ImageHDU(dq, name='DQ')]).writeto(path, overwrite=True)


@pytest.fixture(scope='session')
def rc_fixture(tmp_path_factory):
    """
    Writes synthetic SCI (two rolls, with planet) and REF files, one frame
    per visit; returns their paths and true star positions.
    """
    d = tmp_path_factory.mktemp('rc_data')
    rng = np.random.default_rng(0)
    sci, ref, star_xy = [], [], {}
    for i, roll in enumerate(SCI_ROLLS):
        p = str(d / f'rc_sci_roll{i}.fits')
        star_xy[p] = np.add(STAR_XY, SCI_OFFSETS[i])
        _write_rc_file(p, roll, f'000010010010010010{i}', rng, psfref=False, planet=True, star_xy=star_xy[p])
        sci.append(p)
    for i in range(NREF):
        p = str(d / f'rc_ref_{i}.fits')
        star_xy[p] = np.add(STAR_XY, REF_OFFSETS[i])
        _write_rc_file(p, 0.0, f'000010010010020010{i}', rng, psfref=True, planet=False, star_xy=star_xy[p],
                       bad_pixel=BAD_PIXEL if i == NREF - 1 else None)
        ref.append(p)
    return {'sci': sci, 'ref': ref, 'star_xy': star_xy, 'dir': str(d)}

import numpy as np
from astropy.io import fits

import conftest as c


def test_import_winnie():
    import winnie
    assert hasattr(winnie, 'SpaceRDI')


def test_fixture_files(rc_fixture):
    assert len(rc_fixture['sci']) == len(c.SCI_ROLLS)
    assert len(rc_fixture['ref']) == c.NREF
    with fits.open(rc_fixture['sci'][0]) as hdul:
        assert hdul[0].header['TELESCOP'] == 'ROMAN'
        assert hdul[1].data.shape == (c.NY, c.NX)
        assert hdul['ERR'].data.shape == (1, c.NY, c.NX)
        assert hdul['DQ'].data.shape == (c.NY, c.NX)


def test_fixture_planet_position(rc_fixture):
    # Planet should sit at the expected roll-dependent position in each SCI file.
    for f, roll in zip(rc_fixture['sci'], c.SCI_ROLLS):
        diff = fits.getdata(f, 1) - c._gauss(*rc_fixture['star_xy'][f], fwhm=3.0)
        y, x = np.unravel_index(np.argmax(diff), diff.shape)
        xp, yp = c.planet_xy(roll, rc_fixture['star_xy'][f])
        assert abs(x - xp) <= 1 and abs(y - yp) <= 1

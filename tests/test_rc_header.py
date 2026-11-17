import numpy as np
import pytest
from astropy.io import fits

import conftest as c
from winnie.roman import RCDatabase, read_rc_header


def _headers(f):
    return fits.getheader(f, 0), fits.getheader(f, 1)


def test_sci_fields(rc_fixture):
    for i, (f, roll) in enumerate(zip(rc_fixture['sci'], c.SCI_ROLLS)):
        info = read_rc_header(*_headers(f))
        assert info['posang'] == roll
        assert np.allclose(info['c_star'], c.STAR_XY)
        assert np.allclose(info['c_coron'], info['c_star'])
        assert info['c_coron'] is not info['c_star']
        assert np.isclose(info['pxscale'], c.PXSCALE)
        assert info['wavel'] == 0.575
        assert info['filter'] == '1F'
        assert info['mask'] == 'HLC12_C2R1'
        assert info['lyot'] == 'NFOV'
        assert info['date'] == '2027-01-01T00:00:00'
        assert info['visit'] == f'000010010010010010{i}'
        assert info['psfref'] is False


def test_ref_psfref(rc_fixture):
    info = read_rc_header(*_headers(rc_fixture['ref'][0]))
    assert info['psfref'] is True


def test_roll_fallback(rc_fixture):
    h0, h1 = _headers(rc_fixture['sci'][0])
    del h0['PA_APER']
    h0['ROLL'] = 7.5
    assert read_rc_header(h0, h1)['posang'] == 7.5


def test_errors(rc_fixture):
    h0, h1 = _headers(rc_fixture['sci'][0])

    h1_l1 = h1.copy()
    del h1_l1['STARLOCX']
    h1_l1['DATALVL'] = 'L1'
    with pytest.raises(ValueError, match='L1'):
        read_rc_header(h0, h1_l1)

    h0_noroll = h0.copy()
    del h0_noroll['PA_APER']
    with pytest.raises(ValueError):
        read_rc_header(h0_noroll, h1)

    h1_band = h1.copy()
    h1_band['CFAMNAME'] = '9F'
    with pytest.raises(ValueError):
        read_rc_header(h0, h1_band)


def test_database_from_headers(rc_fixture, tmp_path):
    db = RCDatabase(str(tmp_path / 'out'))
    name = db.add_concat(rc_fixture['sci'], rc_fixture['ref'])
    tab = db.obs[name]
    assert np.isclose(tab['PIXSCALE'][0], c.PXSCALE)
    assert tab['CWAVEL'][0] == 0.575

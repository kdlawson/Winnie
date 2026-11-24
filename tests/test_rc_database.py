import os

import numpy as np
import pytest

import conftest as c
from winnie.roman import RCDatabase


@pytest.fixture
def db(rc_fixture, tmp_path):
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat(rc_fixture['sci'], rc_fixture['ref'], pxscale=c.PXSCALE, cwavel=0.575)
    return db


def test_output_dir(db):
    assert os.path.isdir(db.output_dir)
    assert db.output_dir.endswith(os.sep)
    assert db.red == {}


def test_table_contents(db, rc_fixture):
    tab = db.obs['RC_0']
    assert len(tab) == len(rc_fixture['sci']) + len(rc_fixture['ref'])
    assert np.sum(tab['TYPE'] == 'SCI') == len(rc_fixture['sci'])
    assert np.sum(tab['TYPE'] == 'REF') == len(rc_fixture['ref'])
    for col in ['TYPE', 'FITSFILE', 'PIXSCALE', 'CWAVEL', 'BLURFWHM', 'EXP_TYPE', 'INSTRUME', 'MASKFILE']:
        assert col in tab.colnames
    assert all(os.path.isabs(f) and os.path.isfile(f) for f in tab['FITSFILE'])
    assert tab['PIXSCALE'][0] == c.PXSCALE
    assert tab['EXP_TYPE'][0] == 'RC_CORON'


def test_access_patterns(db):
    # SpaceRDI.load_concat accepts an int index or the concat name.
    assert list(db.obs.keys())[0] == 'RC_0'


def test_multiple_concats(db, rc_fixture):
    name = db.add_concat(rc_fixture['sci'][:1], [], pxscale=c.PXSCALE, cwavel=0.575)
    assert name == 'RC_1'
    assert len(db.obs[name]) == 1


def test_errors(db, rc_fixture):
    with pytest.raises(ValueError):
        db.add_concat([], rc_fixture['ref'], pxscale=c.PXSCALE, cwavel=0.575)
    with pytest.raises(ValueError):
        db.add_concat(['nope.fits'], [], pxscale=c.PXSCALE, cwavel=0.575)
    with pytest.raises(ValueError):
        db.add_concat(rc_fixture['sci'], [], pxscale=c.PXSCALE, cwavel=0.575, concat='RC_0')


def test_spacerdi_init_accepts_rcdatabase(db):
    from winnie import SpaceRDI
    rdi = SpaceRDI(database=db, verbose=False)
    assert rdi.output_dir.startswith(db.output_dir)

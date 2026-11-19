import numpy as np
import pytest

import conftest as c
from winnie import RCRDI
from winnie.roman import RCDatabase


def _centroid(im, xy, hw=5):
    x0, y0 = np.round(xy).astype(int)
    box = np.nan_to_num(im[y0-hw:y0+hw+1, x0-hw:x0+hw+1])
    y, x = np.mgrid[y0-hw:y0+hw+1, x0-hw:x0+hw+1]
    return np.array([np.sum(x*box), np.sum(y*box)]) / np.sum(box)


@pytest.fixture
def db(rc_fixture, tmp_path):
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat(rc_fixture['sci'], rc_fixture['ref'])
    return db


def _load(db, align=True, pad_data=None):
    rdi = RCRDI(db, pad_data=pad_data, verbose=False)
    rdi.load_concat(0, align=align)
    return rdi


def test_shapes_and_metadata(db):
    rdi = _load(db)
    assert rdi.concat == 'RC_0'
    assert rdi.imcube_sci.shape == (len(c.SCI_ROLLS), c.NY, c.NX)
    assert rdi.imcube_ref.shape == (c.NREF, c.NY, c.NX)
    assert rdi.errcube_sci.shape == rdi.imcube_sci.shape
    assert rdi.errcube_ref.shape == rdi.imcube_ref.shape
    assert np.allclose(rdi._posangs_sci, np.mod(c.SCI_ROLLS, 360))
    assert len(set(rdi._visit_ids_sci)) == len(c.SCI_ROLLS)
    assert np.isclose(rdi.pxscale.value, c.PXSCALE)
    assert np.isclose(rdi._lam.value, 0.575)
    # lambda/D at 0.575 um over 2.36 m is 50.3 mas, 2.31 px at 21.8 mas/px.
    assert np.isclose(rdi._fwhm, 2.306, atol=0.01)
    assert rdi.filt == '1F' and rdi._image_mask == 'HLC12_C2R1' and rdi._pupil_mask == 'NFOV'
    assert rdi.siaf_aper is None
    assert rdi.output_ext == 'rdi_psfsub'
    assert rdi.data_ext == 'roll0'


def test_aligned_to_first_sci(db, rc_fixture):
    rdi = _load(db)
    assert np.allclose(rdi.c_star, rc_fixture['star_xy'][rc_fixture['sci'][0]])
    for im in [*rdi.imcube_sci, *rdi.imcube_ref]:
        assert np.allclose(_centroid(im, rdi.c_star), rdi.c_star, atol=0.05)
    assert np.allclose(rdi.c_coron_sci, rdi.c_star)
    assert np.allclose(rdi.c_coron_ref, rdi.c_star)


def test_no_align(db, rc_fixture):
    rdi = _load(db, align=False)
    truth = np.array([rc_fixture['star_xy'][f] for f in [*rc_fixture['sci'], *rc_fixture['ref']]])
    assert np.allclose(np.concatenate([rdi.c_coron_sci, rdi.c_coron_ref]), truth)
    for im, xy in zip([*rdi.imcube_sci, *rdi.imcube_ref], truth):
        assert np.allclose(_centroid(im, xy), xy, atol=0.05)


def test_planet_position(db):
    rdi = _load(db)
    ref = rdi.imcube_ref[0] / c.REF_SCALE
    for im, roll in zip(rdi.imcube_sci, c.SCI_ROLLS):
        xp, yp = c.planet_xy(roll, rdi.c_star)
        assert np.allclose(_centroid(im - ref, (xp, yp), hw=2), (xp, yp), atol=0.1)


def test_bad_pixel(db):
    rdi = _load(db)
    y, x = c.BAD_PIXEL
    assert np.isnan(rdi.imcube_ref[-1, y, x])
    assert np.isnan(rdi.errcube_ref[-1, y, x])
    assert not np.any(rdi.optzones[..., y, x])


def test_padding(db):
    rdi = _load(db, pad_data='auto')
    assert rdi.ny > c.NY and rdi.nx > c.NX
    assert rdi.errcube_sci.shape == rdi.imcube_sci.shape
    assert np.allclose(rdi.c_coron_sci, rdi.c_star)


def test_data_ext_corgidrp_name(rc_fixture, tmp_path):
    import shutil
    f = str(tmp_path / 'cgi_0000100100100100100_20270101t0000000_l3_.fits')
    shutil.copy(rc_fixture['sci'][0], f)
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat([f], [])
    assert RCRDI(db, verbose=False).data_ext == 'l3'


@pytest.fixture
def multi_frame_db(tmp_path):
    """Three frames per SCI visit and two per REF visit, each its own file (as corgidrp writes L3)."""
    rng = np.random.default_rng(1)
    sci, ref = [], []
    for i, roll in enumerate(c.SCI_ROLLS):
        for k in range(3):
            p = str(tmp_path / f'sci{i}_{k}.fits')
            c._write_rc_file(p, roll, f'000010010010010010{i}', rng, planet=True,
                             star_xy=np.add(c.STAR_XY, c.SCI_OFFSETS[i]) + 0.1*k)
            sci.append(p)
    for i in range(c.NREF):
        for k in range(2):
            p = str(tmp_path / f'ref{i}_{k}.fits')
            c._write_rc_file(p, 0.0, f'000010010010020010{i}', rng, psfref=True,
                             star_xy=np.add(c.STAR_XY, c.REF_OFFSETS[i]) - 0.1*k)
            ref.append(p)
    db = RCDatabase(str(tmp_path / 'out'))
    db.add_concat(sci, ref)
    return db


def test_combine_frames(multi_frame_db):
    rdi = RCRDI(multi_frame_db, pad_data=None, verbose=False)
    rdi.load_concat(0)
    assert rdi.imcube_sci.shape == (len(c.SCI_ROLLS), c.NY, c.NX)
    assert rdi.imcube_ref.shape == (c.NREF, c.NY, c.NX)
    assert len(set(rdi._visit_ids_sci)) == len(c.SCI_ROLLS) and len(rdi._files_sci) == len(c.SCI_ROLLS)
    assert np.allclose(rdi._posangs_sci, np.mod(c.SCI_ROLLS, 360))
    # median_combine ERR from the 1e-4 per-frame ERR: 3 frames (SCI) and 2 frames (REF), away from shifted edges.
    inner = (slice(None), slice(5, -5), slice(5, -5))
    assert np.allclose(rdi.errcube_sci[inner], np.sqrt(np.pi*7/12)*1e-4/np.sqrt(3), rtol=1e-3)
    assert np.allclose(rdi.errcube_ref[inner], 1e-4/np.sqrt(2), rtol=1e-3)
    for im in [*rdi.imcube_sci, *rdi.imcube_ref]:
        assert np.allclose(_centroid(im, rdi.c_star), rdi.c_star, atol=0.05)

    frames = RCRDI(multi_frame_db, pad_data=None, verbose=False)
    frames.load_concat(0, combine_frames=False)
    assert frames.imcube_sci.shape == (3*len(c.SCI_ROLLS), c.NY, c.NX)
    assert frames.imcube_ref.shape == (2*c.NREF, c.NY, c.NX)


def test_jackknife_multi_frame(multi_frame_db):
    rdi = RCRDI(multi_frame_db, pad_data=None, verbose=False)
    rdi.load_concat(0)
    inds, rescube, errcube = rdi.jackknife_references(show_progress=False)
    assert rescube.shape == (c.NREF, len(c.SCI_ROLLS), c.NY, c.NX)

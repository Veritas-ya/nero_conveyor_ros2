import pytest

from nero_core.two_second.feedback_gates import GripConfirmation


def test_shorter_dwell_still_requires_fresh_force_and_width_feedback():
    grip=GripConfirmation(.1,.5,1.,dwell_s=.05)
    assert not grip.observe(.03,1.,.99)  # before actual command
    assert not grip.observe(.03,1.,1.01)
    assert not grip.observe(.03,1.,1.01)  # repeated packet is not elapsed evidence
    assert not grip.observe(.03,1.,1.059)
    assert grip.observe(.03,1.,1.06)
    assert not grip.observe(.03,.1,1.07)  # force loss resets the dwell
    assert not grip.observe(.03,1.,1.08)
    assert not grip.observe(.03,1.,1.129)
    assert grip.observe(.03,1.,1.13)


def test_original_confirmation_default_is_preserved():
    grip=GripConfirmation(.1,.5,1.)
    assert not grip.observe(.03,1.,1.01)
    assert not grip.observe(.03,1.,1.06)
    assert grip.observe(.03,1.,1.11)


@pytest.mark.parametrize('dwell',[0.,-.1,float('nan'),float('inf')])
def test_invalid_dwell_rejected(dwell):
    with pytest.raises(ValueError):
        GripConfirmation(.1,.5,1.,dwell_s=dwell)

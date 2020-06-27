from cnp.helpers import Helper


def test_scale_shift_uniform():
    a = -1
    b = 1
    size = (1, 1)

    instance = Helper.scale_shift_uniform(a, b, *size)

    assert (instance.item() > a)
    assert (instance.item() < b)

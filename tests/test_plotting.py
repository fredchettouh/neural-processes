from cnp.plotting import get_contxt_coordinates, get_colour_based_idx
import torch


def test_get_contxt_coordinates():
    contxt = torch.tensor([0, 1, 2, 3])
    rows, cols = get_contxt_coordinates(contxt, 2)

    assert(rows[0] == 0 and rows[2] == 1)
    assert(cols[0] == 0 and cols[2] == 0)

def test_get_colour_based_idx():
    img = torch.tensor([0, 0, 1, 0.2]).reshape(2, 2)
    rows = torch.tensor([0, 1])
    cols = torch.tensor([0, 1])
    white_idx, black_idx = get_colour_based_idx(rows, cols, img)

    assert(white_idx[0] == 1 and len(white_idx) ==1 )
    assert (black_idx[0] == 0 and len(black_idx) == 1)





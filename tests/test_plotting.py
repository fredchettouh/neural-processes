from cnp.plotting import get_contxt_coordinates, get_colour_based_idx, Plotter
import torch


def test_get_contxt_coordinates():
    contxt = torch.tensor([0, 1, 2, 3])
    rows, cols = get_contxt_coordinates(contxt, 2)

    assert (rows[0] == 0 and rows[2] == 1)
    assert (cols[0] == 0 and cols[2] == 0)


def test_get_colour_based_idx():
    img = torch.tensor([0, 0, 1, 0.2]).reshape(2, 2)
    rows = torch.tensor([0, 1])
    cols = torch.tensor([0, 1])
    white_idx, black_idx = get_colour_based_idx(rows, cols, img)

    assert (white_idx[0] == 1 and len(white_idx) == 1)
    assert (black_idx[0] == 0 and len(black_idx) == 1)


def test_plot_training_progress():
    with open('tests/fixtures/train_loss.txt') as file:
        train_loss = [float(line) for line in file.readlines()]
    with open('tests/fixtures/vali_loss.txt') as file:
        vali_loss = [float(line) for line in file.readlines()]

    Plotter.plot_training_progress(
        training_losses=train_loss,
        vali_losses=vali_loss,
        interval=1000)


def test_plot_context_target_1d():
    contxt_idx = torch.randperm(400)[:10]
    xvalues = torch.normal(0, 1, (1, 400, 1))
    target_x = xvalues
    funcvalues = xvalues**2
    target_y = funcvalues
    mu = xvalues
    cov_matrix = torch.normal(0,0.001, (1, 400, 1))
    Plotter.plot_context_target_1d(
        contxt_idx=contxt_idx,
        xvalues=xvalues,
        funcvalues=funcvalues,
        target_y=target_y,
        target_x=target_x,
        mu=mu,
        cov_matrix=cov_matrix)


def test_paint_prediction_greyscale():
    mu = torch.rand((28 * 28))[None, :]
    width = 28
    height = 28
    Plotter.paint_prediction_greyscale(mu=mu, width=width, height=height)


def test_paint_groundtruth_greyscale():
    func_x = torch.rand((28 * 28))[None, :]
    width = 28
    height = 28
    Plotter.paint_groundtruth_greyscale(
        func_x=func_x, width=width, height=height)

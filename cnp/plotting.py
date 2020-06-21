import matplotlib.pyplot as plt
import numpy as np
from .helpers import Helper
import torch


def get_contxt_coordinates(contxt_points, denominator):
    rows = [int(idx / denominator) for idx in contxt_points]
    columns = [(idx % denominator).item() for idx in contxt_points]

    return torch.tensor(rows), torch.tensor(columns)


def get_colour_based_idx(rows, columns, image):
    white_idx = []
    black_idx = []
    for index in range(len(rows)):
        if image[
            int(rows[index].item()),
            int(columns[rows[index].item()])
        ] > 0:
            white_idx.append(index)
        else:
            black_idx.append(index)
    return white_idx, black_idx


class Plotter:

    @staticmethod
    def plot_training_progress(training_losses, vali_losses, interval=1):
        title = "Development of training and validation loss"
        xlabel = "Epoch"
        ylabel = "Negative log probabability "

        xvalues = np.arange(0, len(training_losses), interval)
        plt.plot(xvalues[1:], training_losses[::interval][1:],
                 label='training loss')
        plt.plot(xvalues[1:], vali_losses[1:], label='validation loss')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        plt.close()

    @staticmethod
    def plot_context_target_1d(contxt_idx, xvalues, funcvalues, target_y,
                               target_x, mu, cov_matrix):
        """plots the validation run, i.e. the true function, context points,
        mean function and uncertainty
        It makes also sure that the targetpoints and predictions are ordered
        for propper plotting
         """
        # batch_size = xvalues.shape[0]
        # idx_to_plot = np.random.randint(0, batch_size)
        # simply taking the last from the validation batch so show
        context_y_plot = funcvalues[-1, contxt_idx, :].flatten().cpu()
        context_x_plot = xvalues[-1, contxt_idx, :].flatten().cpu()
        y_plot = target_y[-1].flatten().cpu().numpy()
        x_plot = target_x[-1].flatten().cpu().numpy()
        var_plot = cov_matrix[-1].flatten().cpu().numpy()
        mu_plot = mu[-1].flatten().cpu().numpy()

        x_plot, y_plot, mu_plot, var_plot = Helper.sort_arrays(
            x_plot,
            y_plot,
            mu_plot,
            var_plot)

        plt.scatter(x_plot, y_plot, color='red')
        plt.plot(x_plot, mu_plot, color='blue')
        plt.scatter(context_x_plot, context_y_plot, color='black')

        plt.fill_between(x_plot, y1=mu_plot + var_plot,
                         y2=mu_plot - var_plot, alpha=0.2)
        plt.show()
        plt.close()

    @staticmethod
    def paint_contxt_greyscale(contxt, func_x, width, height):
        img = func_x[-1].reshape(width, height).cpu()

        denominator = height
        rows, columns = get_contxt_coordinates(contxt, denominator)
        get_colour_based_idx(rows, columns, img)
        white_idx, black_idx = get_colour_based_idx(rows, columns, img)
        background = torch.zeros((width, height))

        plt.imshow(background)
        plt.plot(rows[white_idx], columns[white_idx], 'ws', markeredgewidth=2,
                 markersize=3)
        plt.plot(rows[black_idx], columns[black_idx], 'ks', markeredgewidth=2,
                 markersize=3)

        plt.title('Context points for prediction')
        plt.show()
        plt.close()

    @staticmethod
    def paint_prediction_greyscale(mu, width, height):
        mu_plot = mu[-1].cpu().numpy()
        img = mu_plot.reshape(width, height)
        plt.imshow(img, cmap='Greys_r')
        plt.title('Prediction')
        plt.show()
        plt.close()

    @staticmethod
    def paint_groundtruth_greyscale(func_x, width, height):
        groundtruth = func_x[-1].cpu()
        img = groundtruth.reshape(width, height)
        plt.imshow(img, cmap='Greys_r')
        plt.title('Ground truth')
        plt.show()
        plt.close()

    @staticmethod
    def paint_greyscale_images_wrapper(contxt, func_x, mu, width, height):
        Plotter.paint_contxt_greyscale(contxt, func_x, width, height)
        Plotter.paint_prediction_greyscale(mu, width, height)
        Plotter.paint_groundtruth_greyscale(func_x, width, height)

import torch
from torch.nn.functional import softplus
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data


class Helper:

    @staticmethod
    def scale_shift_uniform(a=0, b=1, *size):
        return torch.rand(size=size) * (a - b) + b

    @staticmethod
    def create_loader(datagenerator_instance, num_instances, noise, length_scale, gamma, batch_size):
        x_values, func_x = datagenerator_instance.generate_curves(num_instances, noise, length_scale, gamma)
        func_x = Helper.list_np_to_sensor(func_x)
        x_values = x_values.repeat(func_x.shape[0], 1, 1)
        dataset = data.TensorDataset(x_values, func_x)
        dataloader = data.DataLoader(dataset, batch_size=batch_size)
        return dataloader

    @staticmethod
    def list_np_to_sensor(list_of_arrays, stack=True):
        if stack:
            return torch.stack([array for array in list_of_arrays])
        else:
            return list_of_arrays

    @staticmethod
    def transform_var(var_tensor):

        """This function takes a learned variance tensor and transforms
        it following the methodology in Empirical Evaluation of Neural Process Objectives.
        This ensures that the covariance matrix is positive definite and a multivariate
        Gaussian can be constructed.
        Next it pads the diagonal with zeroes to create a covariance matrix for sampling.
       """
        transformed_variance = 0.1 + 0.9 * softplus(var_tensor)

        return transformed_variance


class Plotter:

    @staticmethod
    def plot_training_progress(training_losses, vali_losses, interval=1):

        title = "Development of training and validation loss"
        xlabel = "Epoch"
        ylabel = "Negative log probabability "
        xvalues = np.arange(0, len(training_losses), interval)
        plt.plot(xvalues, training_losses[::interval], label='training loss')
        plt.plot(xvalues, vali_losses, label='validation loss')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()
        plt.close()


class HyperParam:

    def train_evaluate(parameterization):
        trainer = Experiment(**parameterization)
        weights = trainer.run_training(trainloader)
        evaluation = trainer.run_test(weights, valiloader)
        return evaluation
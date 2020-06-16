import torch
from torch.nn.functional import softplus
from torch import nn
import os
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import pandas as pd
import json
import collections


class Helper:

    @staticmethod
    def scale_shift_uniform(a=0, b=1, *size):
        return torch.rand(size=size) * (a - b) + b

    @staticmethod
    def read_and_transform(data_path, target_col, train_share=0.8, seed=None):
        np.random.seed(seed)
        df = pd.read_csv(data_path)
        df.rename(columns={target_col: 'target'}, inplace=True)

        random_idx = np.random.permutation(len(df))
        train_len = int(train_share * len(df))
        train = df.iloc[random_idx[:train_len]].reset_index(drop=True)
        vali = df.iloc[random_idx[train_len:]].reset_index(drop=True)

        X_train, y_train = train.drop(labels=['target'], axis=1).to_numpy(
            np.float64), train['target'].to_numpy(np.float64)
        X_vali, y_vali = vali.drop(labels=['target'], axis=1).to_numpy(
            np.float64), vali['target'].to_numpy(np.float64)
        X_train, y_train = torch.from_numpy(X_train).float(), \
                           torch.from_numpy(y_train).float()
        X_vali, y_vali = torch.from_numpy(X_vali).float(), \
                         torch.from_numpy(y_vali).float()
        return X_train, y_train, X_vali, y_vali

    @staticmethod
    def shuffletensor(*args):
        arg_list = []
        for arg in args:
            idx = torch.randperm(arg.shape[0])
            if arg.dim() == 1:
                arg = arg[idx]
            else:
                arg = arg[idx, :]
            arg_list.append(arg)
        return arg_list

    @staticmethod
    def sort_arrays(prin_array, *args):
        order_permutation = prin_array.argsort()
        prin_array = prin_array[order_permutation]
        sorted_arrays = [prin_array]
        for arg in args:
            sorted_arrays.append(arg[order_permutation])
        return sorted_arrays

    @staticmethod
    def create_loader(x_values, func_x, batch_size):
        dataset = data.TensorDataset(x_values, func_x)
        dataloader = data.DataLoader(dataset, batch_size=batch_size)
        return dataloader

    @staticmethod
    def list_np_to_tensor(list_of_arrays, stack=True):
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

    @staticmethod
    def init_weights(model):
        """

        Parameters
        ----------
        model :
        """
        x = 0.5
        if type(model) == nn.Linear:
            nn.init.uniform_(model.weight, -0.05, 0.05)
            nn.init.uniform_(model.bias, -0.05, 0.05)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @staticmethod
    def save_results(directory, experiment_name, args):
        current_date = datetime.today()
        day = str(current_date.day).zfill(2)
        month = str(current_date.month).zfill(2)
        year = str(current_date.year)
        hour = str(current_date.hour).zfill(2)
        minute = str(current_date.minute).zfill(2)
        date_time = f"{year}_{month}_{day}_{hour}_{minute}"

        new_dir_name = os.path.join(directory, f"{experiment_name}_{date_time}")
        os.mkdir(new_dir_name)
        for arg in args:
            if arg[1]:
                if type(arg[1]) == collections.OrderedDict:
                    file_name = f"{new_dir_name}/{arg[0]}"
                    torch.save(arg[1], file_name)
                elif type(arg[1]) == list:
                    file_name = f"{new_dir_name}/{arg[0]}.txt"
                    with open(file_name, 'w') as file:
                        for element in arg[1]:
                            file.write(f"{element}\n")
                elif type(arg[1]) == dict:
                    file_name = f"{new_dir_name}/{arg[0]}.json"
                    with open(file_name, 'w') as file:
                        json.dump(arg[0], file)




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
    def plot_run(contxt_idx, xvalues, funcvalues, target_y,
                 target_x, mu, cov_matrix):
        """plots the validation run, i.e. the true function, context points,
        mean function and uncertainty
        It makes also sure that the targetpoints and predictions are ordered
        for propper plotting
         """

        context_y_plot = funcvalues[:, contxt_idx, :].flatten().cpu()
        context_x_plot = xvalues[:, contxt_idx, :].flatten().cpu()
        y_plot = target_y.flatten().cpu().numpy()
        x_plot = target_x.flatten().cpu().numpy()
        var_plot = cov_matrix.flatten().cpu().numpy()
        mu_plot = mu.flatten().cpu().numpy()
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


class HyperParam:

    def train_evaluate(parameterization):
        trainer = Experiment(**parameterization)
        weights = trainer.run_training(trainloader)
        evaluation = trainer.run_test(weights, valiloader)
        return evaluation

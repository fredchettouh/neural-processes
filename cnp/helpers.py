import torch
from torch.nn.functional import softplus
from torch import nn
import os
import random
from datetime import datetime
import numpy as np
from torch.utils import data
import pandas as pd
import json
import collections
import subprocess


class Helper:

    @staticmethod
    def scale_shift_uniform(a=0, b=1, *size):
        """
        Thus function draws from a uniform distribution and shifts
        the results a new spectrum of a and b
        Parameters
        ----------
        a: int lower bound
        b: int: upper bound
        size: tuple: specifying the size of the tensor

        Returns a tensor of entries drawn uniformly from a,b interval
        -------
        """
        return torch.rand(size=size, dtype=torch.float64) * (a - b) + b

    @staticmethod
    def read_and_transform(data_path, target_col, train_share=0.8, seed=None):
        """
        Reads in a data set and tranforms so that it can be devided into
        different tasks. It is not clear that this works in the setting
        of CNPs
        Parameters
        ----------
        data_path: str: path where to find the data set
        target_col: the column name of the target variable
        train_share: float: the share of data to be used for training
        seed: int: seed for reproduciability

        Returns train and validation data
        -------

        """

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
        """
        Parameters
        ----------
        args: tensor: variable number of tensors to be shuffled

        Returns a list of tensors that have been shuffled
        -------

        """
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
    def sort_arrays(principal_tensor, *args):
        """
        Sorts the principal tensor and accepts a variational number of tensors
        to be sorted using those indexes
        Parameters
        ----------
        principal_tensor: tensor: prinicpal tensor which determines the order
                          all other arrays. Importantly expects the tensor
                          to be of 2D
        args: tensors: tensors to be sorted along the indexes of the principal
                       tensor

        Returns tensors sorted along the principal tensor
        -------
        """
        order_permutation = principal_tensor.argsort()
        prin_array = principal_tensor[order_permutation]
        sorted_arrays = [prin_array]
        for arg in args:
            sorted_arrays.append(arg[order_permutation])
        return sorted_arrays

    @staticmethod
    def create_loader(x_values, func_x, batch_size):
        """
        Parameters
        ----------
        x_values: tensor: input values
        func_x: tensor: targets
        batch_size: int

        Returns a torch.nn.loader object
        -------

        """
        dataset = data.TensorDataset(x_values, func_x)
        dataloader = data.DataLoader(dataset, batch_size=batch_size)
        return dataloader

    @staticmethod
    def list_np_to_tensor(list_of_arrays):
        """
        Takes a list of tensors and stacks them up into a tensor
        Parameters
        ----------
        list_of_arrays: list: list of arrays to be stacked, arrays are assumed
                        to be of 2D dimension.

        Returns a tensors from the different arrays
        -------

        """
        return torch.stack([array for array in list_of_arrays])

    @staticmethod
    def transform_var(var_tensor):
        """
        This function takes a learned variance tensor and transforms
        it following the methodology in Empirical Evaluation of Neural Process
        http://bayesiandeeplearning.org/2018/papers/92.pdf
        Objectives. This ensures that the covariance matrix is positive
        definite and a multivariate Gaussian can be constructed.
        Next it pads the diagonal with zeroes to create a covariance matrix for
        sampling.
        Parameters
        ----------
        var_tensor: tensor: Tensor filled with variance values
        Returns a tensor with the variance transformed as outlined above
        -------
        """
        transformed_variance = 0.1 + 0.9 * softplus(var_tensor)

        return transformed_variance

    @staticmethod
    def init_weights(model):
        """
        Ininitialized the weights of the neural network. Can be made
        deterministic by setting torch.seed() before calling it
        Parameters
        ----------
        model : object: torch neural network object
        Returns the model with its weights initialized.
        """
        if type(model) == nn.Linear:
            nn.init.uniform_(model.weight, -0.05, 0.05)
            nn.init.uniform_(model.bias, -0.05, 0.05)

    @staticmethod
    def set_seed(seed):
        """
        Sets the seed at different levels and makes the behaviour of the model
        deterministic. Not currently used because by doing so the number
        context points is always the same. An evaluation of when to seed and un
        set the seed would first have to be done
        Parameters
        ----------
        seed: int: seed to set
        Returns nothing
        -------

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    @staticmethod
    def get_colab_sytstem_info():
        system_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        formatted_info = system_info.stdout.decode("utf-8")
        return formatted_info

    @staticmethod
    def save_results(directory, experiment_name, args,
                     system_information=None):
        """
        This function takes an arbitrary number arguments creates a new
        directory under a specified name with a time stamp and an
        experiment name. It also saves the config file to a specific locatons
        Parameters
        ----------
        system_information
        system information because it is not constant
        directory: str: directory to save the files in
        experiment_name: str: name of the experiment to sd
        args: *args: objects to be saved for example model parameters,
        list of train losses config file.

        Returns nothing
        -------

        """
        current_date = datetime.today()
        day = str(current_date.day).zfill(2)
        month = str(current_date.month).zfill(2)
        year = str(current_date.year)
        hour = str(current_date.hour).zfill(2)
        minute = str(current_date.minute).zfill(2)
        date_time = f"{year}_{month}_{day}_{hour}_{minute}"

        new_dir_name = os.path.join(
            directory, f"{experiment_name}_{date_time}")
        os.mkdir(new_dir_name)
        print(f"Creating new directory at {new_dir_name}")
        for arg in args:
            if arg[1]:
                print(f'Saving {arg[0]}')
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
                        json.dump(arg[1], file)
        if system_information:
            print('Saving snapshot of the sytems')
            file_name = f"{new_dir_name}/system_info.txt"
            with open(file_name, 'w') as file:
                for line in system_information:
                    file.write(line)



# class HyperParam:
#
#     def train_evaluate(parameterization):
#         trainer = Experiment(**parameterization)
#         weights = trainer.run_training(trainloader)
#         evaluation = trainer.run_test(weights, valiloader)
#         return evaluation

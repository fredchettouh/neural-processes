# pytorch imports
import torch
from torch import nn
from torch import optim
# from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# custom imports - WE SHOULD USE RELATIVE IMPORTS HERE
# I.E. from .networks import Encoder, Decoder
# HOWEVER THIS IS NOT POSSIBLE WITH GOOGLE COLAB

from .networks import Encoder, Decoder
from .helpers import Helper, Plotter
from .datageneration import DataGenerator


def get_sample_indexes(min_contx, max_contx, min_trgts, max_trgts,
                       dim_observation, both=True):
    """Samples number and indexes of context and target points during training and tests
    Parameters
    ----------
    both: boolean: Indicates whether both context and target points are required
    """
    num_contxt = np.random.randint(min_contx, max_contx)
    num_trgts = np.random.randint(num_contxt + min_trgts,
                                  num_contxt + max_trgts)
    trgts_idx = np.random.choice(np.arange(0, dim_observation), num_trgts)
    contxt_idx = trgts_idx[:num_contxt]
    if both:
        return trgts_idx, contxt_idx
    else:
        return np.arange(0, dim_observation), contxt_idx


def select_data(contxt_idx, func_idx, xvalues, funcvalues, batch_size):
    num_contxt, num_trgt = len(contxt_idx), len(func_idx)
    context_x = xvalues[:, contxt_idx, :]
    context_y = funcvalues[:, contxt_idx, :]

    target_y = funcvalues[:, func_idx, :]
    target_x = xvalues[:, func_idx, :]

    # the encoding is stacked to ensure a one dimensional input
    context_x_stacked = context_x.view(batch_size * num_contxt, -1)
    context_y_stacked = context_y.view(batch_size * num_contxt, -1)
    target_x_stacked = target_x.view(batch_size * num_trgt, -1)

    return num_contxt, num_trgt, context_x, context_y, target_x, target_y, \
           context_x_stacked, context_y_stacked, target_x_stacked


def format_encoding(encoding, batch_size, num_contxt, num_trgt):
    encoding = encoding.view(batch_size, num_contxt, -1)
    # averaging the encoding
    encoding_avg = encoding.mean(1)
    # we need to unsqueeze and repeat the embedding
    # because we need to pair it with every target
    encoding_avg = encoding_avg.unsqueeze(1)
    encoding_exp = encoding_avg.repeat(1, num_trgt, 1)

    encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)

    return encoding_stacked


class RegressionTrainer:
    """
    This class orchestrates the training, validation and test of the CNP
    Parameters
    ----------
    n_epochs: int: Number of training iterations to run
    lr: float: Learning rate for the Solver

    min_funcs: int: Minimum number of target values to sample

    max_funcs: int: Minimum number of target values to sample

    max_contx: int: Minimum number of context values to sample

    min_contx: int: Minimum number of context values to sample

    dim_observation: int: Number of observations of each "joint target" that means of each function

    dimx: int: Dimension of each x value

    dimy: int: Dimension of each y value

    dimr: tuple: Dimension of the encoding

    dimout: int: Dimensionality of the ouput of the decoder, e.g. batch_size,1,2 for the one d regression case

    num_layers: int: Dimension of hidden layers

    num_neurons: int: Number of Neurons for each hidden layer

    train_on_gpu: boolean: Indicating whether or not a GPU is available

    print_after: int: Indication of the we want to have a validation run

    generatedata: boolean, optional: If True, data will be generated using the Datagenerator

    datagen_params: Dict, optional: Contains the parameters for data generation
    """

    def __init__(self,
                 n_epochs=2000,
                 lr=0.001,
                 min_funcs=2,
                 max_funcs=10,
                 max_contx=10,
                 min_contx=3,
                 dim_observation=400,
                 dimx=1,
                 dimy=1,
                 dimr=50,
                 dimout=2,
                 num_layers_encoder=3,
                 num_neurons_encoder=128,
                 num_layers_decoder=2,
                 num_neurons_decoder=128,
                 dropout=0,
                 train_on_gpu=False,
                 print_after=2000,
                 generatedata=False,
                 data_as_curve=True,
                 range_x=None):

        super().__init__()

        self._n_epochs = n_epochs
        self._lr = lr
        self._dim_observation = dim_observation
        self._train_on_gpu = train_on_gpu
        self._print_after = print_after
        self._generatedata = generatedata
        self._data_as_curve = data_as_curve
        self._encoder = Encoder(dimx, dimy, dimr, num_layers_encoder,
                                num_neurons_encoder)
        self._decoder = Decoder(dimx, num_neurons_encoder, dimout,
                                num_layers_decoder, num_neurons_decoder,
                                dropout)
        self._sample_specs_kwargs = {
            "min_trgts": min_funcs,
            "max_trgts": max_funcs,
            "max_contx": max_contx,
            "min_contx": min_contx
        }

        if self._train_on_gpu:
            self._encoder.cuda()
            self._decoder.cuda()

        if self._generatedata:
            self._datagenerator = DataGenerator(xdim=dimx, ydim=dimy,
                                                range_x=range_x,
                                                steps=dim_observation)

    def _prep_data(self, xvalues, funcvalues, training=True):

        if training:
            func_idx, contxt_idx = get_sample_indexes(
                **self._sample_specs_kwargs,
                dim_observation=self._dim_observation)
        else:
            func_idx, contxt_idx = get_sample_indexes(
                **self._sample_specs_kwargs,
                dim_observation=self._dim_observation,
                both=False)
        batch_size = xvalues.shape[0]

        num_contxt, num_trgt, context_x, context_y, target_x, target_y, \
        context_x_stacked, context_y_stacked, target_x_stacked = select_data(
            contxt_idx, func_idx, xvalues,
            funcvalues, batch_size)

        return num_contxt, num_trgt, context_x, context_y, target_x, target_y, \
               context_x_stacked, context_y_stacked, target_x_stacked, \
               batch_size, func_idx, contxt_idx

    def _network_pass(self, context_x_stacked, context_y_stacked,
                      target_x_stacked, batch_size, num_trgt, num_contxt):

        # running the context values through the encoding
        encoding = self._encoder(context_x_stacked, context_y_stacked)

        encoding_stacked = format_encoding(encoding, batch_size, num_contxt,
                                           num_trgt)
        decoding = self._decoder(target_x_stacked, encoding_stacked)
        decoding_rshp = decoding.view(batch_size, num_trgt, -1)

        mu = decoding_rshp[:, :, 0].unsqueeze(-1)
        sigma = decoding_rshp[:, :, 1].unsqueeze(-1)

        # transforming the variance to ensure that it forms a positive
        # definite covariance matrix
        sigma_transformed = Helper.transform_var(sigma)
        distribution = Normal(loc=mu, scale=sigma_transformed)

        return mu, sigma_transformed, distribution

    def _validation_run(self, current_epoch, plotting, valiloader=None):

        self._encoder.eval()
        self._decoder.eval()

        running_vali_loss = 0

        with torch.no_grad():

            for xvalues, funcvalues in valiloader:

                if self._train_on_gpu:
                    xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                if self._data_as_curve:
                    num_contxt, num_trgt, context_x, context_y, target_x, \
                    target_y, context_x_stacked, context_y_stacked, \
                    target_x_stacked, batch_size, func_idx, \
                    contxt_idx = self._prep_data(
                        xvalues, funcvalues, training=False)

                if not self._generatedata:
                    xvalues = xvalues.unsqueeze(0)
                    funcvalues = funcvalues[None, :, None]

                    num_contxt, num_trgt, context_x, context_y, target_x, \
                    target_y, context_x_stacked, context_y_stacked, \
                    target_x_stacked, batch_size, func_idx, \
                    contxt_idx = self._prep_data(
                        xvalues, funcvalues, training=False)

                mu, sigma_transformed, distribution = self._network_pass(
                    context_x_stacked, context_y_stacked,
                    target_x_stacked, batch_size, num_trgt, num_contxt)

                vali_loss = distribution.log_prob(target_y)
                vali_loss = -torch.mean(vali_loss)
                running_vali_loss += vali_loss.item()
            else:
                mean_vali_loss = running_vali_loss / len(valiloader)
                print(
                    f' Validation loss after {current_epoch} equals {mean_vali_loss}')
                if plotting:
                    Plotter.plot_run(batch_size, contxt_idx, xvalues,
                                     funcvalues, target_y, target_x, mu,
                                     sigma_transformed)
            return mean_vali_loss

    def run_training(self, data_path=None, target_col=None, train_share=None,
                     num_instances_train=None, num_instances_vali=None,
                     noise=None, length_scale=None, gamma=None,
                     batch_size_train=None, batch_size_vali=None,
                     plotting=False, ):
        """This function performs one training run
        Parameters
        ----------
        trainloader: torch.utils.data.DataLoader, optional: iterable object that holds the data in batch sizes

        valiloader: torch.utils.data.DataLoader, optional: iterable object that holds validation data

        plotting: boolean, optional: indicating if progress should be plotted

        batchsize: int, optional: indicating if progress should be plotted
        """
        self._encoder.train()
        self._decoder.train()

        if not self._generatedata:
            X_train, y_train, X_vali, y_vali = Helper.read_and_transform(
                data_path, target_col, train_share)

        optimizer = optim.Adam(self._decoder.parameters())
        mean_epoch_loss = []
        mean_vali_loss = []
        for epoch in tqdm(range(self._n_epochs), total=self._n_epochs):

            if self._generatedata:  # generate data on the fly for every epoch

                X_train, y_train = self._datagenerator.generate_curves(
                    num_instances_train, noise,
                    length_scale, gamma)
                y_train = Helper.list_np_to_sensor.__func__(y_train)
                X_train = X_train.repeat(y_train.shape[0], 1, 1)

                trainloader = Helper.create_loader(
                    X_train, y_train, batch_size_train)

                X_vali, y_vali = self._datagenerator.generate_curves(
                    num_instances_vali,
                    noise, length_scale, gamma)
                y_vali = Helper.list_np_to_sensor.__func__(y_vali)
                X_vali = X_vali.repeat(y_vali.shape[0], 1, 1)

                valiloader = Helper.create_loader(
                    X_vali, y_vali, batch_size_vali)

            #  get sample indexes

            else:
                train = Helper.shuffletensor(X_train, y_train)
                X_train, y_train = train[0], train[1]

                trainloader = Helper.create_loader(
                    X_train, y_train, batch_size_train)
                valiloader = Helper.create_loader(
                    X_vali, y_vali, batch_size_vali)

            running_loss = 0

            for xvalues, funcvalues in trainloader:
                if not self._generatedata:
                    xvalues = xvalues.unsqueeze(0)
                    funcvalues = funcvalues[None, :, None]

                if self._train_on_gpu:
                    xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                optimizer.zero_grad()

                num_contxt, num_trgt, context_x, context_y, target_x, \
                target_y, context_x_stacked, context_y_stacked, \
                target_x_stacked, batch_size, func_idx, \
                contxt_idx = self._prep_data(xvalues,
                                             funcvalues,
                                             training=True)

                mu, sigma_transformed, distribution = self._network_pass(
                    context_x_stacked, context_y_stacked,
                    target_x_stacked, batch_size, num_trgt, num_contxt)

                loss = distribution.log_prob(target_y)
                loss = -torch.mean(loss)
                running_loss += loss

                loss.backward()
                optimizer.step()

            else:
                mean_epoch_loss.append(running_loss / len(trainloader))

                if epoch % self._print_after == 0:
                    print(f'Mean loss at epoch {epoch} : {mean_epoch_loss[-1]}')
                    if valiloader:
                        mean_vali_loss.append(
                            self._validation_run(epoch, plotting, valiloader))
                        self._encoder.train(), self._decoder.train()
        if plotting:
            Plotter.plot_training_progress(mean_epoch_loss, mean_vali_loss,
                                           interval=self._print_after)

        return self._decoder.state_dict()

    def run_test(self, state_dict, testloader, plotting=False):
        """This function performs one test run
                Parameters
                ----------
                testloader: torch.utils.data.DataLoader: iterable object that holds validation data

                state_dict: dictionary: pytorch dictionary to load weights from
        """
        running_mse = 0
        # state_dict = torch.load(file_path_weights)
        self._decoder.load_state_dict(state_dict)

        self._encoder.eval()
        self._decoder.eval()

        with torch.no_grad():

            for xvalues, funcvalues in testloader:

                batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution = self._prep_data(
                    xvalues, funcvalues, training=False)
                mse = ((mu - target_y) ** 2).mean(1).mean(0)
                running_mse += mse.item()
                if plotting:
                    self.plot_run(batch_size, contxt_idx, xvalues, funcvalues,
                                  target_y, target_x, mu,
                                  sigma_transformed)
            else:
                test_set_mse = running_mse / len(testloader)
                return test_set_mse

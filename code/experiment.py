# pytorch imports
import torch
from torch import nn
from torch import optim
# from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# custom imports

from networks import Encoder, Decoder
from helpers import Helper, Plotter


class Experiment(nn.Module):

    """
    This class orchestrates the training, validation and test of the CNP

    Parameters
    ----------

    n_epochs: int
        Number of training iterations to run

    lr: float
        Learning rate for the Solver

    min_funcs: int
        Minimum number of target values to sample

    max_funcs: int
        Minimum number of target values to sample

    max_contx: int
        Minimum number of context values to sample

    min_contx: int
        Minimum number of context values to sample

    dim_observation: int
        Number of observations of each "joint target" that means of each function

    dimx: int
        Dimension of each x value

    dimy: int
        Dimension of each y value

    dimr: tuple
        Dimension of the encoding

    dimout: int
        Dimensionality of the ouput of the decoder, e.g. batch_size,1,2 for the one d regression case


    num_layers : int
        Dimension of hidden layers

    num_neurons: int

        Number of Neurons for each hidden layer

    train_on_gpu: boolean

        Indicating whether or not a GPU is available

    print_after: int
        Indication of the we want to have a validation run

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
                 train_on_gpu=False,
                 print_after=2000):
        super().__init__()

        self._n_epochs = n_epochs
        self._lr = lr
        self._min_trgts = min_funcs
        self._max_trgts = max_funcs
        self._max_contx = max_contx
        self._min_contx = min_contx
        self._dim_observation = dim_observation
        self._train_on_gpu = train_on_gpu
        self._print_after = print_after

        self._encoder = Encoder(dimx, dimy, dimr, num_layers_encoder, num_neurons_encoder)
        self._decoder = Decoder(dimx, num_neurons_encoder, dimout, num_layers_decoder, num_neurons_decoder)

        if self._train_on_gpu:
            self._encoder.cuda()
            self._decoder.cuda()



    def _get_sample_indexes(self, both=True):

        """Samples number and indexes of context and target points during training and test

        Parameters
        ----------

        both: boolean
            Indicates whether both context and target points are required
        """
        num_contxt = np.random.randint(self._min_contx, self._max_contx)
        num_trgts = np.random.randint(num_contxt + self._min_trgts, num_contxt + self._max_trgts)
        trgts_idx = np.random.choice(np.arange(0, self._dim_observation), num_trgts)
        contxt_idx = trgts_idx[:num_contxt]
        if both:
            return trgts_idx, contxt_idx
        else:
            return np.arange(0, self._dim_observation), contxt_idx

    def _prep_data(self, xvalues, funcvalues, training=True):

        """For every batch this function is called to prepare the data, i.e.
        sampling context and target points, shaping them appropriately and
        passing them through the encoder and decoder
        Parameters
        ----------

        xvalues: tensor
            (batch_size,self._dim_observation, self._xdim) that stores all the points
            to sample from in this batch

        funcvalues: tensor
            (batch_size,self._dim_observation, self._ydim) that stores all the points
            to sample from in this batch

        training: boolean
            indicates whether this is a training pass or a validation pass

        returns:
        batch_size,
        target_x,
        target_y,
        context_x,
        contxt_idx,
        context_y,
        mu,
        sigma_transformed,
        distribution

        """

        if self._train_on_gpu:
            xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()
        if training:
            func_idx, contxt_idx = self._get_sample_indexes()
        else:
            func_idx, contxt_idx = self._get_sample_indexes(both=False)
        batch_size = xvalues.shape[0]

        num_contxt, num_trgt = len(contxt_idx), len(func_idx)
        context_y = funcvalues[:, contxt_idx, :]
        context_x = xvalues[:, contxt_idx, :]

        target_y = funcvalues[:, func_idx, :]
        target_x = xvalues[:, func_idx, :]

        # the encoding is stacked to ensure a one dimensional input
        context_y_stacked = context_y.view(batch_size * num_contxt, -1)
        context_x_stacked = context_x.view(batch_size * num_contxt, -1)

        # running the context values through the encoding
        encoding = self._encoder(context_x_stacked, context_y_stacked)
        encoding = encoding.view(batch_size, num_contxt, -1)
        # averaging the encoding
        encoding_avg = encoding.mean(1)
        # we need to unsqueeze and repeat the embedding
        # because we need to pair it with every target
        encoding_avg = encoding_avg.unsqueeze(1)
        encoding_exp = encoding_avg.repeat(1, num_trgt, 1)

        encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)
        target_x_stacked = target_x.view(batch_size * num_trgt, -1)
        decoding = self._decoder(target_x_stacked, encoding_stacked)
        decoding_rshp = decoding.view(batch_size, num_trgt, -1)

        mu, sigma = decoding_rshp[:, :, 0].unsqueeze(-1), decoding_rshp[:, :, 1].unsqueeze(-1)
        # transforming the variance to ensure that it forms a positive definite covariance matrix
        sigma_transformed = Helper.transform_var(sigma)
        distribution = Normal(loc=mu, scale=sigma_transformed)

        return batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution

    def plot_run(self, batch_size, contxt_idx, xvalues, funcvalues, target_y, target_x, mu, cov_matrix):

        """plots the validation run, i.e. the true function, context points, mean function and uncertainty"""

        random_function = np.random.randint(0, batch_size)
        context_y_plot = funcvalues[random_function, contxt_idx, :].flatten().cpu()
        context_x_plot = xvalues[random_function, contxt_idx, :].flatten().cpu()
        y_plot = target_y[random_function, :, :].flatten().cpu()
        x_plot = target_x[random_function, :, :].flatten().cpu()
        mu_plot = mu[random_function, :, :].flatten().cpu()
        var_plot = cov_matrix[random_function, :, :].flatten().cpu()
        plt.scatter(x_plot, y_plot, color='red')
        plt.scatter(context_x_plot, context_y_plot, color='black')
        plt.plot(x_plot, mu_plot, color='blue')
        plt.fill_between(x_plot, y1=mu_plot + var_plot, y2=mu_plot - var_plot, alpha=0.2)
        plt.show()
        plt.close()

    def _validation_run(self, valiloader, current_epoch, plotting):

        self._encoder.eval()
        self._decoder.eval()

        running_vali_loss = 0
        self.eval()

        with torch.no_grad():

            for xvalues, funcvalues in valiloader:

                batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution = self._prep_data(
                    xvalues, funcvalues, training=False)
                vali_loss = distribution.log_prob(target_y)
                vali_loss = -torch.mean(vali_loss)
                running_vali_loss += vali_loss.item()
            else:
                mean_vali_loss = running_vali_loss / len(valiloader)
                print(f' Validation loss after {current_epoch} equals {mean_vali_loss}')
                if plotting:
                    print(plotting)
                    self.plot_run(batch_size, contxt_idx, xvalues, funcvalues, target_y, target_x, mu,
                                  sigma_transformed)
            return mean_vali_loss

    def run_training(self, trainloader, valiloader=None, plotting=False):

        """This function performs one training run
        Parameters
        ----------

        trainloader: torch.utils.data.DataLoader
            iterable object that holds the data in batch sizes

        valiloader: torch.utils.data.DataLoader
            iterable object that holds validation data

        plotting: boolean
            indicating if progress should be plotted
        """

        self._encoder.train()
        self._decoder.train()

        optimizer = optim.Adam(self._decoder.parameters())
        mean_epoch_loss = []
        mean_vali_loss = []
        for epoch in tqdm(range(self._n_epochs), total=self._n_epochs):
            # for epoch in range(self._n_epochs):
            running_loss = 0
            #         get sample indexes
            for xvalues, funcvalues in trainloader:
                optimizer.zero_grad()
                batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution = self._prep_data(
                    xvalues,
                    funcvalues,
                    training=True)
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
                        print(plotting)
                        mean_vali_loss.append(self._validation_run(valiloader, epoch, plotting))
                        self._encoder.train(), self._decoder.train()
        if plotting:
            Plotter.plot_training_progress(mean_epoch_loss, mean_vali_loss, interval=self._print_after)

        return self._decoder.state_dict()

    def run_test(self, state_dict, testloader, plotting=False):

        """This function performs one test run
                Parameters
                ----------

                testloader: torch.utils.data.DataLoader
                    iterable object that holds validation data

                state_dict: dictionary
                    pytorch dictionary to load weights from
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
                    self.plot_run(batch_size, contxt_idx, xvalues, funcvalues, target_y, target_x, mu, sigma_transformed)

            else:
                test_set_mse = running_mse / len(testloader)
                return test_set_mse



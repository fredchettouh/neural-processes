# pytorch imports
import torch
from torch import optim
from tqdm import tqdm
from importlib import import_module
import numpy as np
from copy import copy

# custom imports - WE SHOULD USE RELATIVE IMPORTS HERE
# I.E. from .networks import Encoder, Decoder
# HOWEVER THIS IS NOT POSSIBLE WITH GOOGLE COLAB

from .helpers import Helper, Plotter
from .cnp import RegressionCNP


# TODO: optimizer has to become variable
# else I cannot pass different aggregators to it


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

    dim_observation: int: Number of observations of each "joint target" that
    =means of each function

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
                 data_kwargs,
                 cnp,
                 dimx,
                 range_x,
                 dim_observation,
                 n_epochs=2000,
                 lr=0.001,
                 train_on_gpu=False,
                 seed=None):

        super().__init__()

        self._n_epochs = n_epochs
        self._lr = lr
        self._train_on_gpu = train_on_gpu
        self._seed = seed
        self._cnp = cnp

        self._cnp.encoder.apply(Helper.init_weights)
        self._cnp.decoder.apply(Helper.init_weights)

        if self._train_on_gpu:
            self._cnp.encoder.cuda()
            self._cnp.decoder.cuda()
        data_kwargs = copy(data_kwargs)
        datagenerator = data_kwargs.pop('datagenerator')
        if datagenerator:
            package_name, method_name = datagenerator.rsplit('.', 1)
            package = import_module(package_name)
            method = getattr(package, method_name)

            self._datagenerator = method(
                xdim=dimx,
                range_x=range_x,
                steps=dim_observation)
            self.data_kwargs = data_kwargs

        else:
            self._datagenerator = datagenerator
            package_name, method_name = \
                'cnp.datageneration.DataGenerator'.rsplit('.', 1)
            package = import_module(package_name)
            method = getattr(package, method_name)
            self._base_datagenerator = method(
                xdim=dimx,
                range_x=range_x,
                steps=dim_observation)

            self._X_train, self._y_train, self._X_vali, self._y_vali = \
                Helper.read_and_transform(
                    data_path=data_kwargs['data_path'],
                    target_col=data_kwargs['target_col'],
                    train_share=data_kwargs['train_share'],
                    seed=data_kwargs['seed']
                )

    def _validation_run(self, current_epoch, print_after, valiloader=None):

        self._cnp.encoder.eval()
        self._cnp.decoder.eval()
        if self._cnp.aggregator:
            self._cnp.aggregator.eval()

        running_vali_loss = 0

        with torch.no_grad():

            for xvalues, funcvalues in valiloader:

                if self._train_on_gpu:
                    xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                if not self._datagenerator:
                    xvalues = xvalues.unsqueeze(0)
                    funcvalues = funcvalues[None, :, None]

                contxt_idx, xvalues, funcvalues, target_y, target_x, mu, \
                sigma_transformed, distribution = \
                    self._cnp.prep_and_pass(
                        xvalues, funcvalues, training=False,
                    )

                vali_loss = distribution.log_prob(target_y)
                vali_loss = -torch.mean(vali_loss)
                running_vali_loss += vali_loss.item()

            else:

                # all values (x, y, loss, are the values of the last iteration
                # Therefore this function is also plotted

                mean_vali_loss = running_vali_loss / len(valiloader)
                print(
                    f'Mean Validation loss after {current_epoch} equals\
                     {round(mean_vali_loss, 3)}\n')

                print(f'Validation loss for the function plotted: \
                {round(vali_loss.item(), 3)}')

                if print_after:
                    Plotter.plot_run(contxt_idx, xvalues,
                                     funcvalues, target_y, target_x, mu,
                                     sigma_transformed)
            return mean_vali_loss

    def run_training(self, print_after=None, batch_size_train=None,
                     batch_size_vali=None):

        """This function performs one training run
        Parameters
        ----------

        print_after: boolean, optional: indicating if progress should be plotted

        batch_size_train: batch size of data

        batch_size_vali: batch size of data

        **kwargs: dict, takes key value pair data depening on wether data
        is generated on the fly or read in.

        """
        if self._seed:
            Helper.set_seed(self._seed)

        self._cnp.encoder.train()
        self._cnp.decoder.train()
        if self._cnp.aggregator:
            self._cnp.aggregator.eval()

        # Todo optimizier should be passed as an argument to the trainer
        if self._cnp.aggregator:

            optimizer = optim.Adam(
                list(self._cnp.encoder.parameters()) +
                list(self._cnp.aggregator.parameters()) +
                list(self._cnp.decoder.parameters()),
                lr=self._lr)
        else:
            optimizer = optim.Adam(
                list(self._cnp.encoder.parameters()) +
                list(self._cnp.decoder.parameters()),
                lr=self._lr)

        mean_epoch_loss = []
        mean_vali_loss = []

        for epoch in tqdm(range(self._n_epochs), total=self._n_epochs):

            if self._datagenerator:  # generate data on the fly for every epoch

                trainloader = self._datagenerator.generate_loader_on_fly(
                    batch_size_train, self.data_kwargs, purpose='train')

                valiloader = self._datagenerator.generate_loader_on_fly(
                    batch_size_vali, self.data_kwargs, purpose='vali')

            else:
                trainloader = \
                    self._base_datagenerator.generate_from_single_function(
                        self._X_train, self._y_train, batch_size_train,
                        shuffle=True)
                valiloader = \
                    self._base_datagenerator.generate_from_single_function(
                        self._X_vali, self._y_vali, batch_size_vali,
                        shuffle=False)

            running_loss = 0

            for xvalues, funcvalues in trainloader:
                if not self._datagenerator:
                    xvalues = xvalues.unsqueeze(0)
                    funcvalues = funcvalues[None, :, None]

                if self._train_on_gpu:
                    xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                optimizer.zero_grad()

                contxt_idx, xvalues, funcvalues, target_y, target_x, mu, \
                sigma_transformed, distribution = \
                    self._cnp.prep_and_pass(
                        xvalues, funcvalues, training=True)

                loss = distribution.log_prob(target_y)
                loss = -torch.mean(loss)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            else:

                mean_epoch_loss.append(running_loss / len(trainloader))
                if epoch % print_after == 0:
                    print(f'Mean training loss at epoch {epoch} : \
                        {round(mean_epoch_loss[-1], 3)}')
                    if valiloader:
                        mean_vali_loss.append(
                            self._validation_run(
                                epoch, print_after, valiloader))
                        self._cnp.encoder.train()
                        self._cnp.decoder.train()
                        if self._cnp.aggregator:
                            self._cnp.aggregator.train()
        if print_after:
            Plotter.plot_training_progress(mean_epoch_loss, mean_vali_loss,
                                           interval=print_after)
            encoder_state_dict = self._cnp.encoder.state_dict()
            decoder_state_dict = self._cnp.decoder.state_dict()
            if self._cnp.aggregator:
                aggregator_state_dict = self._cnp.aggregator.state_dict()
            else:
                aggregator_state_dict = None

        return encoder_state_dict, decoder_state_dict, aggregator_state_dict,\
               mean_epoch_loss, mean_vali_loss


    def run_test(
            self, encoder_state_dict, decoder_state_dict, aggregator_state_dict,
            batch_size_test, X_test=None, y_test=None, plotting=True):
        """This function performs one test run
                Parameters
                ----------
                y_test
                X_test
                aggregator_state_dict
                decoder_state_dict
                encoder_state_dict
                state_dict_decoder
                state_dict_encoder
                plotting
                batch_size_test
                kwargs
                testloader: torch.utils.data.DataLoader: iterable object that
                 holds validation data

                state_dict: dictionary: pytorch dictionary to load weights from
        """
        np.random.seed(self._seed)
        self._cnp.encoder.load_state_dict(encoder_state_dict)
        self._cnp.decoder.load_state_dict(decoder_state_dict)
        self._cnp.encoder.eval()
        self._cnp.decoder.eval()

        if aggregator_state_dict:
            self._cnp.aggregator.load_state_dict(decoder_state_dict)
            self._cnp.aggregator.eval()

        if self._datagenerator:  # generate data on the fly for every epoch

            testloader = self._datagenerator.generate_loader_on_fly(
                batch_size_test, self.data_kwargs, purpose='test')

        else:
            testloader = self._datagenerator.generate_from_single_function(
                X_test, y_test, batch_size_test,
                shuffle=True)

        running_mse = 0
        with torch.no_grad():

            for xvalues, funcvalues in testloader:

                contxt_idx, xvalues, funcvalues, target_y, target_x, mu, \
                sigma_transformed, distribution = \
                    self._cnp.prep_and_pass(
                        xvalues, funcvalues, training=False)

                mse = ((mu - target_y) ** 2).mean(1).mean(0)
                running_mse += mse.item()
                if plotting:
                    Plotter.plot_run(
                        contxt_idx, xvalues, funcvalues, target_y, target_x, mu,
                        sigma_transformed)
            else:
                test_set_mse = running_mse / len(testloader)
                return test_set_mse

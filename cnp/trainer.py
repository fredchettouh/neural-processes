# pytorch imports
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from importlib import import_module
import numpy as np

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
                 num_layers_aggr=None,
                 num_neurons_aggr=None,
                 dropout=0,
                 train_on_gpu=False,
                 print_after=2000,
                 datagenerator=None,
                 range_x=None,
                 seed=None):

        super().__init__()

        self._n_epochs = n_epochs
        self._lr = lr
        self._train_on_gpu = train_on_gpu
        self._print_after = print_after
        self._datagenerator = datagenerator
        self._seed = seed

        aggregation_kwargs = {
            "insize": max_contx,
            "dimout": 1,
            "num_layers": num_layers_aggr,
            "num_neurons": num_neurons_aggr,
            "dropout": dropout
        }

        self._cnp = RegressionCNP(
            min_funcs=min_funcs,
            max_funcs=max_funcs,
            max_contx=max_contx,
            min_contx=min_contx,
            dimx=dimx,
            dimy=dimy,
            dimr=dimr,
            num_layers_encoder=num_layers_encoder,
            num_neurons_encoder=num_neurons_encoder,
            dimout=dimout,
            num_layers_decoder=num_layers_decoder,
            num_neurons_decoder=num_neurons_decoder,
            aggregation_kwargs=aggregation_kwargs,
            dropout=dropout)

        Helper.set_seed(self._seed)
        self._cnp.encoder.apply(Helper.init_weights)
        self._cnp.decoder.apply(Helper.init_weights)

        if self._train_on_gpu:
            self._cnp.encoder.cuda()
            self._cnp.decoder.cuda()

        if datagenerator:
            package_name, method_name = datagenerator.rsplit('.', 1)
            package = import_module(package_name)
            method = getattr(package, method_name)

            self._datagenerator = method(
                xdim=dimx,
                range_x=range_x,
                steps=dim_observation)

    def _validation_run(self, current_epoch, plotting, valiloader=None):
        Helper.set_seed(self._seed)

        self._cnp.encoder.eval()
        self._cnp.decoder.eval()

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
                        xvalues, funcvalues, training=False)

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

                if plotting:
                    Plotter.plot_run(contxt_idx, xvalues,
                                     funcvalues, target_y, target_x, mu,
                                     sigma_transformed)
            return mean_vali_loss

    def run_training(self, plotting=False, batch_size_train=None,
                     batch_size_vali=None, kwargs=None):
        """This function performs one training run
        Parameters
        ----------

        plotting: boolean, optional: indicating if progress should be plotted

        batch_size_train: batch size of data

        batch_size_vali: batch size of data

        **kwargs: dict, takes key value pair data depening on wether data
        is generated on the fly or read in.

        """
        Helper.set_seed(self._seed)

        self._cnp.encoder.train()
        self._cnp.decoder.train()

        if not self._datagenerator:
            X_train, y_train, X_vali, y_vali = Helper.read_and_transform(
                data_path=kwargs['data_path'],
                target_col=kwargs['target_col'],
                train_share=kwargs['train_share'],
                seed=kwargs['seed']
            )

        if self._cnp.aggregator:

            optimizer = optim.Adam(list(self._cnp.encoder.parameters()) +
                                   list(self._cnp.aggregator.parameters()) +
                                   list(self._cnp.decoder.parameters()))
        else:
            optimizer = optim.Adam(list(self._cnp.encoder.parameters()) +
                                   list(self._cnp.decoder.parameters()))

        mean_epoch_loss = []
        mean_vali_loss = []

        for epoch in tqdm(range(self._n_epochs), total=self._n_epochs):

            # TODO:this part should be happening in the data generation part

            if self._datagenerator:  # generate data on the fly for every epoch

                kwargs['purpose'] = 'train'

                X_train, y_train = self._datagenerator.generate_curves(
                    **kwargs)

                y_train = Helper.list_np_to_tensor(y_train)
                X_train = X_train.repeat(y_train.shape[0], 1, 1)

                trainloader = Helper.create_loader(
                    X_train, y_train, batch_size_train)

                kwargs['purpose'] = 'vali'

                X_vali, y_vali = self._datagenerator.generate_curves(**kwargs)

                y_vali = Helper.list_np_to_tensor(y_vali)
                X_vali = X_vali.repeat(y_vali.shape[0], 1, 1)

                valiloader = Helper.create_loader(
                    X_vali, y_vali, batch_size_vali)

            else:

                # TODO this should happen in data generation
                train = Helper.shuffletensor(X_train, y_train)
                X_train, y_train = train[0], train[1]

                trainloader = Helper.create_loader(
                    X_train, y_train, batch_size_train)
                valiloader = Helper.create_loader(
                    X_vali, y_vali, batch_size_vali)

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
                if epoch % self._print_after == 0:
                    print(f'Mean training loss at epoch {epoch} : \
                        {round(mean_epoch_loss[-1], 3)}')
                    if valiloader:
                        mean_vali_loss.append(
                            self._validation_run(epoch, plotting, valiloader))
                        self._cnp.encoder.train(), self._cnp.decoder.train()
        if plotting:
            Plotter.plot_training_progress(mean_epoch_loss, mean_vali_loss,
                                           interval=self._print_after)

        return self._cnp.encoder.state_dict(), self._cnp.decoder.state_dict()

    def run_test(self, state_dict_encoder, state_dict_decoder,
                 batch_size_test, plotting, kwargs):
        """This function performs one test run
                Parameters
                ----------
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
        self._cnp.encoder.load_state_dict(state_dict_encoder)
        self._cnp.decoder.load_state_dict(state_dict_decoder)

        self._cnp.encoder.eval()
        self._cnp.decoder.eval()

        if self._datagenerator:  # generate data on the fly for every epoch

            kwargs['purpose'] = 'test'
            X_test, y_test = self._datagenerator.generate_curves(
                **kwargs)

            y_test = Helper.list_np_to_tensor(y_test)
            X_test = X_test.repeat(y_test.shape[0], 1, 1)

            testloader = Helper.create_loader(
                X_test, y_test, batch_size_test)
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

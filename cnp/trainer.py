# pytorch imports
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from importlib import import_module

# custom imports - WE SHOULD USE RELATIVE IMPORTS HERE
# I.E. from .networks import Encoder, Decoder
# HOWEVER THIS IS NOT POSSIBLE WITH GOOGLE COLAB

from .helpers import Helper, Plotter
from .cnp import RegressionCNP


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
                 datagenerator=None,
                 range_x=None):

        super().__init__()

        self._n_epochs = n_epochs
        self._lr = lr
        self._train_on_gpu = train_on_gpu
        self._print_after = print_after
        self._datagenerator = datagenerator

        self._cnp = RegressionCNP(
            min_funcs, max_funcs, max_contx, min_contx,
            dimx, dimy, dimr, num_layers_encoder, num_neurons_encoder, dimout,
            num_layers_decoder, num_neurons_decoder, dropout)

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
                mean_vali_loss = running_vali_loss / len(valiloader)
                print(
                    f' Validation loss after {current_epoch} equals\
                     {mean_vali_loss}')
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
        self._cnp.encoder.train()
        self._cnp.decoder.train()

        if not self._datagenerator:
            X_train, y_train, X_vali, y_vali = Helper.read_and_transform(
                data_path=kwargs['data_path'],
                target_col=kwargs['target_col'],
                train_share=kwargs['train_share'],
                seed=kwargs['seed']
            )

        optimizer = optim.Adam(self._cnp.decoder.parameters())
        mean_epoch_loss = []
        mean_vali_loss = []

        for epoch in tqdm(range(self._n_epochs), total=self._n_epochs):

            # TODO:this part should be happening in the data generation part

            if self._datagenerator:  # generate data on the fly for every epoch

                kwargs['train'] = True

                X_train, y_train = self._datagenerator.generate_curves(
                    **kwargs)

                y_train = Helper.list_np_to_sensor(y_train)
                X_train = X_train.repeat(y_train.shape[0], 1, 1)

                trainloader = Helper.create_loader(
                    X_train, y_train, batch_size_train)

                kwargs['train'] = False

                X_vali, y_vali = self._datagenerator.generate_curves(**kwargs)

                y_vali = Helper.list_np_to_sensor(y_vali)
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
                        xvalues, funcvalues, training=False)

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
                        self._cnp.encoder.train(), self._cnp.decoder.train()
        if plotting:
            Plotter.plot_training_progress(mean_epoch_loss, mean_vali_loss,
                                           interval=self._print_after)

        return self._cnp.decoder.state_dict()

    # TODO this needs to be updated
    def run_test(self, state_dict, testloader, plotting=False):
        """This function performs one test run
                Parameters
                ----------
                testloader: torch.utils.data.DataLoader: iterable object that
                 holds validation data

                state_dict: dictionary: pytorch dictionary to load weights from
        """
        running_mse = 0
        # state_dict = torch.load(file_path_weights)
        self._cnp.decoder.load_state_dict(state_dict)

        self._cnp.encoder.eval()
        self._cnp.decoder.eval()

        with torch.no_grad():

            for xvalues, funcvalues in testloader:

                batch_size, target_x, target_y, context_x, contxt_idx,\
                    context_y, mu, sigma_transformed, distribution = \
                    self._cnp.prep_data(xvalues, funcvalues, training=False)
                mse = ((mu - target_y) ** 2).mean(1).mean(0)
                running_mse += mse.item()
                if plotting:
                    self.plot_run(contxt_idx, xvalues, funcvalues,
                                  target_y, target_x, mu,
                                  sigma_transformed)
            else:
                test_set_mse = running_mse / len(testloader)
                return test_set_mse

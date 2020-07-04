# pytorch imports
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from importlib import import_module
from copy import copy

from .helpers import Helper
from .plotting import Plotter


# TODO: optimizer has to become variable
# else I cannot pass different aggregators to it

class RegressionTrainer:
    """
    This class orchestrates the training, validation and test of the
    few shot conditional neural process
    Parameters
    ----------
    data_kwargs: dict: key value arguments to be passed to data generation
    cnp: object: The conditional neural process instant to use for training
    n_epochs: int: Number of training iterations to run
    lr: float: Learning rate for the Solver
    train_on_gpu: boolean: Indicating whether or not a GPU is available
    seed: int: the seed to use for reproduciability

    """

    def __init__(self,
                 data_kwargs,
                 cnp,
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

            if isinstance(self._cnp.aggregator, nn.Module):
                self._cnp.aggregator.apply(Helper.init_weights)
                self._cnp.aggregator.cuda()

        data_kwargs = copy(data_kwargs)
        datagenerator = data_kwargs.pop('datagenerator')

        if datagenerator:
            # data generator is imported dynamically
            package_name, method_name = datagenerator.rsplit('.', 1)
            package = import_module(package_name)
            method = getattr(package, method_name)

            init_kwargs = data_kwargs.pop('init_kwargs')
            self._datagenerator = method(
                **init_kwargs)
            self.data_kwargs = data_kwargs

        else:
            # data is read in from existing file and transformed
            self._datagenerator = datagenerator
            package_name, method_name = \
                'cnp.datageneration.DataGenerator'.rsplit('.', 1)
            init_kwargs = data_kwargs.pop('init_kwargs')
            package = import_module(package_name)
            method = getattr(package, method_name)

            self._base_datagenerator = method(**init_kwargs)

            self._X_train, self._y_train, self._X_vali, self._y_vali = \
                Helper.read_and_transform(
                    data_path=data_kwargs['data_path'],
                    target_col=data_kwargs['target_col'],
                    train_share=data_kwargs['train_share'],
                    seed=data_kwargs['seed']
                )

    def _validation_run(self, current_epoch, plot_mode=None, valiloader=None):
        """

        Parameters
        ----------
        Performs a validation run and plotts the proceddings if indicated
        Returns the mean validation loss which is the negative
        log probability, i.e. the probability that the targets where produced
        by the distribution parameterized by the predicted mean and variance.

        current_epoch: int: indicates the epoch the training process is
         currently in
        plot_mode: str: defines the type of plot, i.e. none, 1d regression
        or greyscale image. can be continued to allow for different plotes
        valiloader: object: torch data loader expected to be passed down
        by train_run

        """

        self._cnp.encoder.eval()
        self._cnp.decoder.eval()
        if isinstance(self._cnp.aggregator, nn.Module):
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

                if plot_mode == '1d_regression':
                    Plotter.plot_context_target_1d(
                        contxt_idx, xvalues,
                        funcvalues, target_y, target_x, mu,
                        sigma_transformed)

                    print(f'Validation loss for the function plotted: \
                                    {round(vali_loss.item(), 3)}')

                elif plot_mode == '2d_greyscale':
                    Plotter.paint_greyscale_images_wrapper(
                        contxt_idx, funcvalues, mu, width=28, height=28)
                    print(
                        f'Validation loss for the function plotted: \
                                    {round(vali_loss.item(), 3)}')

            return mean_vali_loss

    def run_training(
            self,
            print_after=None,
            batch_size_train=None,
            plot_mode=None,
            batch_size_vali=None,
            plot_progress=True):

        """
        This function performs the training of the conditional neural process
        This also includes the data generation on the fly or sampling from
        a distribution of tasks (images, different functions etc.)
        Returns the model parameters for the encoder, decoder and aggregator
        as well as the mean losses and training losses
        Parameters
        ----------

        plot_progress: bool: indicates whether or not the training progress
                       shoudl be plotted
        plot_mode: str:
        print_after: boolean, optional: indicating if progress should be
        batch_size_train: batch size of data

        batch_size_vali: batch size of data
        """
        self._cnp.encoder.train()
        self._cnp.decoder.train()
        if isinstance(self._cnp.aggregator, nn.Module):
            self._cnp.aggregator.train()

        # Todo optimizier should be passed as an argument to the trainer
        if isinstance(self._cnp.aggregator, nn.Module):

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
                                epoch, plot_mode, valiloader))
                        self._cnp.encoder.train()
                        self._cnp.decoder.train()
                        if isinstance(self._cnp.aggregator, nn.Module):
                            self._cnp.aggregator.train()
        if plot_progress:
            Plotter.plot_training_progress(
                mean_epoch_loss, mean_vali_loss, interval=print_after)
        encoder_state_dict = self._cnp.encoder.state_dict()
        decoder_state_dict = self._cnp.decoder.state_dict()
        if isinstance(self._cnp.aggregator, nn.Module):
            aggregator_state_dict = self._cnp.aggregator.state_dict()
        else:
            aggregator_state_dict = None

        return encoder_state_dict, decoder_state_dict, aggregator_state_dict,\
            mean_epoch_loss, mean_vali_loss

    def run_test(
            self, encoder_state_dict, decoder_state_dict,
            aggregator_state_dict, batch_size_test, X_test=None, y_test=None,
            plot_mode=None):
        """This function performs one test run
             Parameters
             ----------
             y_test: tensor: targets
             X_test: tensor: inputs
             aggregator_state_dict: object: parameters of the aggregation model
             decoder_state_dict: object: parameters of the decoder model
             encoder_state_dict: object: parameters of the encoder model
             plot_mode: str: indicator of the plotting function should be used
             batch_size_test: int: batch_size of the test set.
        """
        self._cnp.encoder.load_state_dict(encoder_state_dict)
        self._cnp.decoder.load_state_dict(decoder_state_dict)
        self._cnp.encoder.eval()
        self._cnp.decoder.eval()

        if aggregator_state_dict:
            self._cnp.aggregator.load_state_dict(aggregator_state_dict)
            self._cnp.aggregator.eval()
        if self._seed:
            print('seed is set')
            Helper.set_seed(self._seed)

        if self._datagenerator:  # generate data on the fly for every epoch

            testloader = self._datagenerator.generate_loader_on_fly(
                batch_size_test, self.data_kwargs, purpose='test')

        else:
            testloader = self._datagenerator.generate_from_single_function(
                X_test, y_test, batch_size_test,
                shuffle=True)

        running_mse = 0
        task_mses = []
        with torch.no_grad():

            for xvalues, funcvalues in testloader:
                if self._train_on_gpu:
                    xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                contxt_idx, xvalues, funcvalues, target_y, target_x, mu, \
                    sigma_transformed, distribution = \
                    self._cnp.prep_and_pass(
                        xvalues, funcvalues, training=False)

                point_mse = ((mu - target_y) ** 2)
                task_mse = point_mse.mean(1)[:, 0]
                batch_mse = task_mse.mean()
                task_mses.append(mse.item() for mse in task_mse)

                running_mse += batch_mse.item()
                if plot_mode == '1d_regression':
                    Plotter.plot_context_target_1d(
                        contxt_idx, xvalues,
                        funcvalues, target_y, target_x, mu,
                        sigma_transformed)

                elif plot_mode == '2d_greyscale':
                    Plotter.paint_greyscale_images_wrapper(
                        contxt_idx, funcvalues, mu, width=28, height=28)
            else:
                test_set_mse = running_mse / len(testloader)
                flattened_mses = [
                    inner for outer in task_mses for inner in outer
                ]
                return test_set_mse, flattened_mses

from torch import nn
from .networks import Encoder, Decoder, TargetBasedAggregation
from .helpers import Helper
import numpy as np
from torch.distributions.normal import Normal
from importlib import import_module
from copy import copy


def select_data(contxt_idx, func_idx, xvalues, funcvalues, batch_size):
    """
    Uses context indexes and target indexes to select data from available
    inputs and outputs. Also formats the data before returning it
    Returns the number of context points, the number of target points
    the target x and y as well as the stacked context and target.

    Parameters
    ----------
    batch_size: int: number of tasks drawn from distribution of tasks per batch
    funcvalues: tensor: (batch size, num_points ,Y_dim) actual task
                data output
    xvalues: tensor: (batch size, num_points, x_dim) input for task
    func_idx: np.array: (num_targets) indexes for target points
    contxt_idx : np.array: (num_context) indexes for context point
    """

    num_contxt, num_trgt = len(contxt_idx), len(func_idx)
    context_x = xvalues[:, contxt_idx, :]
    context_y = funcvalues[:, contxt_idx, :]

    target_y = funcvalues[:, func_idx, :]
    target_x = xvalues[:, func_idx, :]

    # the encoding is stacked to ensure a one dimensional input
    context_x_stacked = context_x.view(batch_size * num_contxt, -1)
    context_y_stacked = context_y.view(batch_size * num_contxt, -1)
    target_x_stacked = target_x.view(batch_size * num_trgt, -1)
    return num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
        context_y_stacked, target_x_stacked


def get_sample_indexes(
        min_contx, max_contx, min_trgts, max_trgts, dim_observation,
        both=True):
    """
    Samples a random number and indexes for context and target points
    during training and at test time.  Note that the target points always i
    include the context points as per specification of the conditional neural
    process paper.
    Returns context indixes and target indexes at training time
    Parameters
    ----------
    min_contx: int: the minimum number of context points to sample
    max_contx: int: the maximum number of context points to sample
    min_trgts: int: the minimum number of target points to sample
    max_trgts: int: the maximum number of target points to sample
    dim_observation: int: the total number of points to sample indexes from
    both: bool: Indicates whether both context and target points are required
    fix_num_contxt: bool: Indicates if the number of context points should be
                    fixed t every iteration
    """

    if not both:
        trgts_idx = np.arange(0, dim_observation)
        num_contxt = max_contx // 2
        contxt_idx = np.random.permutation(dim_observation)[:num_contxt]

    else:
        num_contxt = np.random.randint(min_contx, max_contx)
        num_trgts = np.random.randint(min_trgts, max_trgts)
        trgts_idx = np.random.choice(
            np.arange(0, dim_observation),
            num_contxt+num_trgts, replace=False)

        contxt_idx = trgts_idx[:num_contxt]

    return trgts_idx, contxt_idx


def format_encoding(encoding, batch_size, num_trgt, repeat=True):
    """
    Utility function that extend takes the aggregated prior knowledge of one
    dimension and repeats it to achieve the original shape of the context input
    Returns a tensor (batch_size * num_context,  dim_x)

    Parameters
    ----------
    repeat: bool: indicates if the context has to be repeated to all target
                  points. This is not the case if attention based attention
                  is used
    num_trgt: int: number of target points needed to associate the duplicate
              the aggregated context (prior knowlegdge)
    batch_size: int: number of tasks in batch
    encoding : tensor: (batch_size, 1, x_dim) the aggregated prior
    """

    # because we need to pair it with every target
    if repeat:
        encoding = encoding.repeat(1, num_trgt, 1)
    encoding_stacked = encoding.view(batch_size * num_trgt, -1)

    return encoding_stacked


class RegressionCNP:
    """
    Class that implements a  Condtional Neural Process for few shot regression.
    It implements the encoder, decoder and if specified the aggregator.
    Also defines all the functions to performa one pass through the CNP
    for instance the sampling and forward methods.

    min_funcs: int: min number of target values
    max_funcs: int: max number of target values
    max_contx: int: max number of context points
    min_contx: int: min number of context points
    fix_num_contxt: bool: indicates whether or not the number of context points
                    should be fixed at every iteration
    dimx: int: dimension of the input values, in 1d regression 1 in grey scale
          pixel regression 2
    dimy: int: output dimesnion usually 1
    dimr: int: dimensionality of the aggregation r
    num_layers_encoder: int: number of layers for the encoder
    num_neurons_encoder: int: number of neurons for each decoder layer
    num_layers_decoder: int: number of layers for the decoder
    num_neurons_decoder: int: number of neurons for the decoder
    dimout: int:
            dimensionality of the final output, usually two (mean, variance)
    aggregation_kwargs: dict: holds all key value arguments for instantiation
                        of the aggregation network if it is wanted
    dropout: float
    Returns a class with encoder and decoder objects
    """

    def __init__(
            self,
            min_funcs,
            max_funcs,
            max_contx,
            min_contx,
            dimx,
            dimy,
            dimr,
            num_layers_encoder,
            num_neurons_encoder,
            dimout,
            num_layers_decoder,
            num_neurons_decoder,
            aggregation_kwargs,
            dropout=0,
            activation='nn.ReLU()',
            batch_norm=False
    ):
        super().__init__()

        self._encoder = Encoder(
            insize=dimx + dimy,
            num_layers=num_layers_encoder,
            num_neurons=num_neurons_encoder,
            dimout=dimr,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm
        )

        self._decoder = Decoder(
            insize=dimx + dimr,
            num_layers=num_layers_decoder,
            num_neurons=num_neurons_decoder,
            dimout=dimout,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm)

        print(self._decoder)
        print(self._encoder)

        aggregation_kwargs = copy(aggregation_kwargs)
        method_name = aggregation_kwargs.pop('aggregator')
        package_name = 'cnp.networks'
        package = import_module(package_name)
        method = getattr(package, method_name)

        self._aggregator = method(**aggregation_kwargs)
        aggregation_kwargs['aggregator'] = method_name
        # else:
        #     self._aggregator = None
        #     self.simple_aggregator_type = aggregation_kwargs[
        #         "simple_aggregator_type"
        #     ]

        print(self._aggregator)

        self._sample_specs_kwargs = {
            "min_trgts": min_funcs,
            "max_trgts": max_funcs,
            "max_contx": max_contx,
            "min_contx": min_contx,
        }

    def prep_data(self, xvalues, funcvalues, training=True,
                  ):
        """
        Takes a input output values from one function (task) and prepares
        the data for being passed through the model, i.e. sampling and
        formatting
        Returns the sampled context and target data as well as some meta
        information such as the number of context points selected etc.
        Parameters
        ---------
        xvalues: tensor: batch_size, steps, xdim
        funcvalues: tensor: batch_size, steps, ydim
        training: bool : indicates if this is a training or validation run.
         During testing we expect the model to return all values as
         target points
        """
        if training:
            func_idx, contxt_idx = get_sample_indexes(
                **self._sample_specs_kwargs,
                dim_observation=xvalues.shape[1],
            )

        else:
            func_idx, contxt_idx = get_sample_indexes(
                **self._sample_specs_kwargs,
                dim_observation=xvalues.shape[1],
                both=False,
            )
        batch_size = xvalues.shape[0]
        num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
            context_y_stacked, target_x_stacked = \
            select_data(
                contxt_idx, func_idx, xvalues, funcvalues, batch_size)

        return num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
            context_y_stacked, target_x_stacked, batch_size, contxt_idx

    def network_pass(
            self, context_x_stacked, context_y_stacked, target_x_stacked,
            batch_size, num_trgt, num_contxt):
        """
        Function that passes the sampled context points though the encoder
        the aggregation process and then through the decoder
        Returns tensor of mean estimates, the variance tensor and the
        distribution object parameterized by them.
        Parameters
        ----------
        num_contxt: number of context points
        num_trgt: int: number of target points
        batch_size: int: number of tasks given per batch
        target_x_stacked: tensor : batch_size * num_target_points, x_dim
        context_y_stacked: batch_size * num_context_points, y_dim
        context_x_stacked : tensor : batch_size * num_context_points, x_dim
        """
        # running the context values through the encoding
        encoding = self._encoder(context_x_stacked, context_y_stacked)

        # shaping it back to batch_view dimensions to decouple it
        # from type of aggregator (i.e. mean, NN, RNN)
        encoding_batch_view = encoding.view(batch_size, num_contxt, -1)

        if isinstance(self._aggregator, TargetBasedAggregation):
            context_x = context_x_stacked.view(batch_size, num_contxt, -1)
            target_x = target_x_stacked.view(batch_size, num_trgt, -1)

            attention_weighted_encdoding = \
                self._aggregator.apply_target_based_attention(
                    context_x, target_x, encoding_batch_view)

            encoding_stacked = format_encoding(
                attention_weighted_encdoding, batch_size, num_trgt,
                repeat=False)

        else:
            if isinstance(self._aggregator, nn.Module):
                aggregated_enconding = self._aggregator(encoding_batch_view)
            else:
                aggregated_enconding = \
                    self._aggregator.simple_aggregation(encoding_batch_view)
            encoding_stacked = format_encoding(
                aggregated_enconding, batch_size, num_trgt, repeat=True)

        decoding = self._decoder(target_x_stacked, encoding_stacked)
        decoding_rshp = decoding.view(batch_size, num_trgt, -1)

        mu = decoding_rshp[:, :, 0].unsqueeze(-1)
        sigma = decoding_rshp[:, :, 1].unsqueeze(-1)

        # transforming the variance to ensure that it forms a positive
        # definite covariance matrix
        sigma_transformed = Helper.transform_var(sigma)
        distribution = Normal(loc=mu, scale=sigma_transformed)

        return mu, sigma_transformed, distribution

    def prep_and_pass(self, xvalues, funcvalues, training=True,
                      ):
        num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
            context_y_stacked, target_x_stacked, batch_size, contxt_idx = \
            self.prep_data(xvalues, funcvalues, training)
        mu, sigma_transformed, distribution = self.network_pass(
            context_x_stacked, context_y_stacked, target_x_stacked,
            batch_size, num_trgt, num_contxt)

        return contxt_idx, xvalues, funcvalues, target_y, target_x, mu, \
            sigma_transformed, distribution

    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def aggregator(self):
        return self._aggregator

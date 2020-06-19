from .networks import Encoder, Decoder, simple_aggregation
from .helpers import Helper
import numpy as np
from torch.distributions.normal import Normal
import torch
from importlib import import_module
from copy import copy


def select_data(contxt_idx, func_idx, xvalues, funcvalues, batch_size):
    """

    Parameters
    ----------
    batch_size:
    funcvalues:
    xvalues:
    func_idx:
    contxt_idx :
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
        min_contx, max_contx, min_trgts, max_trgts, dim_observation, both=True,
        fix_num_contxt=False):
    """Samples number and indexes of context and target points during training
     and tests
    Parameters
    ----------
    fix_num_contxt
    dim_observation
    max_trgts
    min_trgts
    max_contx
    min_contx
    both: boolean: Indicates whether both context and target points are required
    """
    if fix_num_contxt:
        num_contxt = max_contx//2
    else:
        # TODO this is not implmented as in in the paper:
        # num_target = tf.random_uniform(
        # shape=(), minval=2, maxval=self._max_num_context, dtype=tf.int32)
        # In emiel this is implemented without replacement and with num_targets
        # not necceserraily being larger than num_contxt
        num_contxt = np.random.randint(min_contx, max_contx)
    num_trgts = np.random.randint(min_trgts, max_trgts)
    trgts_idx = np.random.choice(
        np.arange(0, dim_observation), num_trgts, replace=False)
    contxt_idx = trgts_idx[:num_contxt]
    if both:
        return trgts_idx, contxt_idx
    else:
        return np.arange(0, dim_observation), contxt_idx


def format_encoding(encoding, batch_size, num_trgt):
    """

    Parameters
    ----------
    embedding_dim
    num_trgt
    batch_size
    encoding :
    """

    # because we need to pair it with every target
    encoding_exp = encoding.repeat(1, num_trgt, 1)
    encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)

    return encoding_stacked


class RegressionCNP:
    def __init__(
            self,
            min_funcs,
            max_funcs,
            max_contx,
            min_contx,
            fix_num_contxt,
            dimx,
            dimy,
            dimr,
            num_layers_encoder,
            num_neurons_encoder,
            dimout,
            num_layers_decoder,
            num_neurons_decoder,
            aggregation_kwargs,
            dropout=0):
        super().__init__()

        self._encoder = Encoder(
            insize=dimx+dimy,
            num_layers=num_layers_encoder,
            num_neurons=num_neurons_encoder,
            dimout=dimr)

        self._decoder = Decoder(
            insize=dimx+dimr,
            num_layers=num_layers_decoder,
            num_neurons=num_neurons_decoder,
            dimout=dimout,
            dropout=dropout)

        # self._decoder = Decoder(dimx, num_neurons_encoder, dimout,
        #                         num_layers_decoder, num_neurons_decoder,
        #                         dropout)

        print(self._decoder)
        print(self._encoder)

        aggregation_kwargs = copy(aggregation_kwargs)
        method_name = aggregation_kwargs.pop('aggregator')
        if method_name:
            package_name = 'cnp.networks'
            package = import_module(package_name)
            method = getattr(package, method_name)

            self._aggregator = method(**aggregation_kwargs)
            aggregation_kwargs['aggregator'] = method_name
        else:
            self._aggregator = None
            self.simple_aggregator_type = aggregation_kwargs[
                "simple_aggregator_type"
            ]

        print(self._aggregator)

        self._sample_specs_kwargs = {
            "min_trgts": min_funcs,
            "max_trgts": max_funcs,
            "max_contx": max_contx,
            "min_contx": min_contx,
            "fix_num_contxt": fix_num_contxt
        }

    def prep_data(self, xvalues, funcvalues, training=True,
                  ):
        """

        Parameters
        ---------
        fix_num_contxt
        training
        funcvalues
        xvalues :
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
            batch_size, num_trgt, num_contxt, hidden=None):
        """

        Parameters
        ----------
        hidden
        num_contxt
        num_trgt
        batch_size
        target_x_stacked
        context_y_stacked
        context_x_stacked :
        """
        # running the context values through the encoding
        encoding = self._encoder(context_x_stacked, context_y_stacked)

        # shaping it back to batch_view dimensions to decouple it
        # from type of aggregator (i.e. mean, NN, RNN)
        encoding_batch_view = encoding.view(batch_size, num_contxt, -1)

        if self._aggregator:
            # todo we need to figureout here what to do with hidden
            aggregated_enconding = self._aggregator(
                encoding_batch_view)
        else:
            aggregated_enconding = simple_aggregation(
                encoding_batch_view, self.simple_aggregator_type)

        encoding_stacked = format_encoding(
            aggregated_enconding, batch_size, num_trgt)

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

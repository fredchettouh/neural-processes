from .networks import Encoder
from .networks import Decoder
from .helpers import Helper
import numpy as np
from torch.distributions.normal import Normal


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


class RegressionCNP:
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
            dropout=0):
        super().__init__()

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

    def _prep_data(self, xvalues, funcvalues, training=True):
        if training:
            func_idx, contxt_idx = get_sample_indexes(
                **self._sample_specs_kwargs,
                dim_observation=xvalues.shape[1])
        else:
            func_idx, contxt_idx = get_sample_indexes(
                **self._sample_specs_kwargs,
                dim_observation=xvalues.shape[1],
                both=False)
        batch_size = xvalues.shape[0]

        num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
            context_y_stacked, target_x_stacked = \
            select_data(
                contxt_idx, func_idx, xvalues, funcvalues, batch_size)

        return num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
            context_y_stacked, target_x_stacked, batch_size, contxt_idx

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

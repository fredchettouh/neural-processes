from cnp.cnp import format_encoding, get_sample_indexes, select_data, \
    RegressionCNP
import torch

import numpy as np

seed = 0


def test_format_encoding():
    aggregated_encoding = torch.randn(64, 1, 128)
    num_contxt = 5
    batch_size = 64

    formatted_encoding = format_encoding(
        aggregated_encoding, batch_size, num_contxt)

    assert (formatted_encoding.shape[0] == batch_size * num_contxt)


# todo: test needs to be written properly once the sampling has been worked out
def test_get_sample_indexes():
    min_contx = 5
    max_contx = 10
    min_trgts = 5
    max_trgts = 10
    dim_observation = 50
    both = True

    trgts_idx, contxt_idx = get_sample_indexes(
        min_contx, max_contx, min_trgts, max_trgts, dim_observation, both)

    len_unique_trgt = len(np.unique(trgts_idx))
    len_unique_con = len(np.unique(contxt_idx))

    assert (len_unique_trgt == len(trgts_idx))
    assert (len_unique_con == len(contxt_idx))

    assert (min_trgts + min_contx <= len_unique_trgt <= max_trgts + max_contx)
    assert (min_contx <= len_unique_con <= max_contx)


def test_select_data():
    # contxt_idx, func_idx, xvalues, funcvalues, batch_size
    for i in range(10):
        steps = np.random.randint(1, 400)
        x_dim = np.random.randint(1, 10)
        batch_size = np.random.randint(1, 20)
        x_values = torch.normal(0, 1, (batch_size, steps, x_dim))
        func_x = torch.normal(0, 1, (batch_size, steps, 1))
        n_context = np.random.randint(1, steps)
        n_targets = np.random.randint(1, steps)
        context_idx = np.random.randint(1, steps, n_context)
        target_idx = np.random.randint(1, steps, n_targets)
        num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
            context_y_stacked, target_x_stacked = \
            select_data(context_idx, target_idx, x_values, func_x, batch_size)

        assert (num_contxt == n_context)
        assert (num_trgt == n_targets)
        batch_size_x, num_x, dim_x = target_x.size()
        batch_size_y, num_y, dim_y = target_y.size()
        assert (batch_size_x == batch_size)
        assert (num_x == n_targets)
        assert (dim_x == x_dim)
        assert (context_x_stacked.shape[0] == batch_size * num_contxt)
        assert (batch_size_y == batch_size)
        assert (num_y == n_targets)
        assert (dim_y == 1)


def test_RegressionCNP():
    for i in range(10):
        batch_size_in = np.random.randint(1, 64)
        x_dim = np.random.randint(1, 10)
        steps = np.random.randint(10, 400)
        y_dim = 1
        r_dim = np.random.randint(1, 128)

        xvalues = torch.normal(0, 1, (batch_size_in, steps, x_dim))
        y_values = torch.normal(0, 1, (batch_size_in, steps, y_dim))

        regressor = RegressionCNP(
            min_funcs=2,
            max_funcs=4,
            max_contx=4,
            min_contx=2,
            dimx=x_dim,
            dimy=1,
            dimr=r_dim,
            num_layers_encoder=2,
            num_neurons_encoder=20,
            dimout=2,
            num_layers_decoder=2,
            num_neurons_decoder=10,
            aggregation_kwargs={
                "aggregator": "GatedMLPAggregator",
                "insize": r_dim,
                "num_layers": 2,
                "num_neurons": 64,
                "dimout": 1
            },
            dropout=0.1
        )

        num_contxt, num_trgt, target_x, target_y, context_x_stacked, \
            context_y_stacked, target_x_stacked, batch_size, contxt_idx = \
            regressor.prep_data(
                xvalues=xvalues, funcvalues=y_values, training=True)

        assert (target_x_stacked.shape[0] == batch_size * num_trgt)
        assert (context_x_stacked.shape[0] == batch_size * num_contxt)
        assert (batch_size == batch_size_in)
        assert (context_y_stacked.shape[0] == batch_size * num_contxt)
        assert (context_x_stacked.shape[1] == x_dim)

    mu, sigma_transformed, distribution = regressor.network_pass(
        context_x_stacked, context_y_stacked, target_x_stacked, batch_size,
        num_trgt, num_contxt)

    assert mu.shape[0] == batch_size_in
    assert sigma_transformed.shape[1] == num_trgt

    contxt_idx, xvalues, funcvalues, target_y, target_x, mu, \
        sigma_transformed, distribution = \
        regressor.prep_and_pass(xvalues, y_values, training=True)

    assert (funcvalues.size() == y_values.size())
    assert mu.shape[0] == batch_size_in

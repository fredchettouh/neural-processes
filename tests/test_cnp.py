from cnp.cnp import format_encoding
# get_sample_indexes
import torch

# import numpy as np

seed = 0


def test_format_encoding():
    aggregated_encoding = torch.randn(64, 1, 128)
    num_contxt = 5
    batch_size = 64

    formatted_encoding = format_encoding(
        aggregated_encoding, batch_size, num_contxt)

    assert (formatted_encoding.shape[0] == batch_size * num_contxt)

# todo: test needs to be written properly once the sampling has been worked out
# def test_get_sample_indexes():
#     min_contx = 5
#     max_contx = 10
#     min_trgts = 5
#     max_trgts = 10
#     dim_observation = 50
#     both = True
#     np.random.RandomState(0)
#     num_contxt_expected = np.random.randint(min_contx, max_contx)
#     num_targt_expected = np.random.randint(num_contxt_expected + min_trgts,
#                                            num_contxt_expected + max_trgts)
#
#     trgts_idx_expected = np.random.choice(
#         np.arange(0, dim_observation), num_targt_expected)
#     contxt_idx_expected = trgts_idx_expected[:num_targt_expected]
#
#     trgts_idx, contxt_idx = get_sample_indexes(
#     min_contx, max_contx, min_trgts, max_trgts, dim_observation, both)
# #
#     assert (trgts_idx == trgts_idx_expected)
#     assert (contxt_idx == contxt_idx_expected)

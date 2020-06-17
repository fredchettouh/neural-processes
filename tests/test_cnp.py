from cnp.cnp import format_encoding, get_sample_indexes
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

    assert(len_unique_trgt >= min_trgts and len_unique_trgt <= max_trgts)
    assert (len_unique_con >= min_trgts and len_unique_con <= max_trgts)

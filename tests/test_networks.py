from cnp.networks import simple_aggregation
import torch


def test_simple_aggregation():
    tensor = torch.Tensor([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    aggregation = simple_aggregation(tensor, 'mean')
    batch_size, embedding_dim, = aggregation.size()
    assert(batch_size == 2)
    assert(embedding_dim == 3)

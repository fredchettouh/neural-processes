from cnp.networks import simple_aggregation, BasicMLP, Encoder, Decoder, \
    BasicMLPAggregator, GatedMLPAggregator
import torch


def test_simple_aggregation():
    tensor = torch.Tensor([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    aggregation = simple_aggregation(tensor, 'mean')
    batch_size, aggregation_dim, embedding_dim, = aggregation.size()
    assert (batch_size == 2)
    assert (aggregation_dim == 1)
    assert (embedding_dim == 3)


def test_BasicMLP():
    mlp = BasicMLP(10, 5, 128, 1, 0)
    children = list(mlp.children())[0]
    assert (len(children) == 9)
    assert (children[0].in_features == 10)
    assert (children[-1].out_features == 1)


def test_Encoder():
    encoder = Encoder(10, 5, 128, 1, 0)
    children = list(encoder.children())[0]
    assert (len(children) == 9)
    assert (children[0].in_features == 10)
    assert (children[-1].out_features == 1)


def test_Decoder():
    decoder = Decoder(10, 5, 128, 1, 0)
    children = list(decoder.children())[0]
    assert (len(children) == 9)
    assert (children[0].in_features == 10)
    assert (children[-1].out_features == 1)


def test_BasicMLPAggregator():
    encoding = torch.randn(64, 5, 128)
    aggr = BasicMLPAggregator(128, 2, 10, 1)

    aggregation = aggr(encoding)

    batch_size, _, dim_encoding = aggregation.size()
    assert (batch_size == 64)
    assert (dim_encoding == 128)

def test_GatedMLPAggregator():
    encoding = torch.randn(64, 5, 128)
    aggr = GatedMLPAggregator(insize=128,
                              num_layers=2,
                              num_neurons=128,
                              dimout=1)

    aggregation = aggr(encoding)

    batch_size, mean_dim, embedding_dim = aggregation.size()
    assert (batch_size == 64)
    assert (mean_dim == 1)
    assert (embedding_dim == 128)

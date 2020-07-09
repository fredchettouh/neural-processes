from cnp.networks import BasicMLP, Encoder, Decoder, \
    BasicMLPAggregator, GatedMLPAggregator, SimpleAggregator,\
    TargetBasedAggregation
import torch


def test_SimpleAggregator():
    tensor = torch.Tensor([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
    aggr = SimpleAggregator(aggregation_operation='mean')
    aggregation = aggr.simple_aggregation(tensor)
    batch_size, aggregation_dim, embedding_dim, = aggregation.size()
    assert (batch_size == 2)
    assert (aggregation_dim == 1)
    assert (embedding_dim == 3)
    aggr = SimpleAggregator(aggregation_operation='max')
    aggregation = aggr.simple_aggregation(tensor)
    batch_size, aggregation_dim, embedding_dim, = aggregation.size()
    assert (batch_size == 2)
    assert (aggregation_dim == 1)
    assert (embedding_dim == 3)
    aggr = SimpleAggregator(aggregation_operation='sum')
    aggregation = aggr.simple_aggregation(tensor)
    batch_size, aggregation_dim, embedding_dim, = aggregation.size()
    assert (batch_size == 2)
    assert (aggregation_dim == 1)
    assert (embedding_dim == 3)


def test_TargetBasedAggregation():
    aggr = TargetBasedAggregation('laplace_kernel')
    targets_x = torch.normal(0, 1, (2, 10, 10))
    context_x = torch.normal(0, 1, (2, 5, 10))
    rep_x = torch.normal(0, 1, (2, 5, 1))
    result = aggr.apply_target_based_attention(context_x, targets_x, rep_x)

    result_bs, result_np, result_dim = result.size()
    assert (result_bs == 2 and result_np == 10 and result_dim == 1)


def test_BasicMLP():
    mlp = BasicMLP(10, 5, 128, 1, 0,
                   activation='nn.ReLU()', batch_norm=True)
    children = list(mlp.children())[0]
    assert (len(children) == 10)
    assert (children[0].in_features == 10)
    assert (children[-1].out_features == 1)


def test_Encoder():
    encoder = Encoder(10, 5, 128, 1, 0,
                      activation='nn.ReLU()', batch_norm=True)
    children = list(encoder.children())[0]
    assert (len(children) == 10)
    assert (children[0].in_features == 10)
    assert (children[-1].out_features == 1)


def test_Decoder():
    decoder = Decoder(10, 5, 128, 1, 0, batch_norm=True)
    children = list(decoder.children())[0]
    assert (len(children) == 10)
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
                              num_neurons=128,
                              dimout=1)

    aggregation = aggr(encoding)

    batch_size, mean_dim, embedding_dim = aggregation.size()
    assert (batch_size == 64)
    assert (mean_dim == 1)
    assert (embedding_dim == 128)

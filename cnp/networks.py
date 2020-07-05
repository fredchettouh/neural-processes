import torch
from torch import nn
from importlib import import_module


def create_linear_layer(
        layer_specs, index, dropout, activation, batch_norm=False):
    """
    Creates a linear layer automatically given specification.
    Returns a list with all object specified for this particular layer
    Parameters
    ----------
    batch_norm: bool: indicating whether batch_normalization should be used
    activation: object: activation function object to be used in this layer
    index: int: Indicates at which layer in the layer architecture
    specification the model currently is

    layer_specs: list: Holds the specification for all layers in the
        architecture

    dropout: float: specifies the dropout probability to be used in the dropout
        layer
    """
    lin_layer = nn.Linear(layer_specs[index], layer_specs[index + 1])
    activation_function = activation
    dropout_layer = nn.Dropout(p=dropout)

    layer_object = [lin_layer, activation_function]

    if batch_norm:
        layer_object.append(nn.BatchNorm1d(layer_specs[index + 1]))

    if dropout:
        layer_object.append(dropout_layer)

    return layer_object


class BasicMLP(nn.Module):

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0,
                 activation='nn.ReLU()', batch_norm=False):
        super().__init__()

        self._insize = insize

        self._dimout = dimout
        self._dropout = dropout

        activation = eval(activation)

        self._hidden_layers = [num_neurons for _ in range(num_layers)]

        _first_layer = [
            nn.Linear(self._insize, self._hidden_layers[0]),
            activation]

        if batch_norm:
            _first_layer.append(nn.BatchNorm1d(self._hidden_layers[0]))
        if dropout:
            _first_layer.append(nn.Dropout(p=dropout))

        _hidden_layers = [
            create_linear_layer(
                self._hidden_layers, i, dropout, activation)
            for i in range(len(self._hidden_layers) - 2)]
        _hidden_layers_flat = [
            element for inner in _hidden_layers for element in inner]

        _last_layer = [
            nn.Linear(self._hidden_layers[-1], self._dimout)]

        self._layers = _first_layer + _hidden_layers_flat + _last_layer

        self._process_input = nn.Sequential(*self._layers)

    @property
    def process_input(self):
        return self._process_input


class Encoder(BasicMLP):
    """
    This class maps each x_i, y_i context point to a representation r_i
    To learn this Representation we are using a Multi Layer Perceptron
    The input shape will be batch_size, num_context_points, x_dim

    The input to the encoder are the value pairs, thus the dimensions are
     Batch_Size, (dimx+dimy). The Pytorch automatically pases the values
     sequentially through the ANN. The last layer will not have an activation
    function because we want the pure represenation.

     Parameters
    ----------
    dimx: int: Dimesion of each x value

    dimy: int: Dimesion of each y value
    dimr: int: Dimension of output representation

    num_layers: int: Dimension of hidden layers

    num_neurons: int: Number of Neurons for each hidden layer

    """

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0,
                 activation='nn.ReLU()', batch_norm=False):
        super().__init__(
            insize, num_layers, num_neurons, dimout, dropout, activation,
            batch_norm)

    def forward(self, x_values, y_values):
        """
        Parameters
        ----------
        x_values: torch.Tensor: Shape (batch_size, dimx)

        y_values: torch.Tensor: Shape (batch_size, dimy)
        """

        input_as_pairs = torch.cat((x_values, y_values), dim=1)
        return self._process_input(input_as_pairs)


class Decoder(BasicMLP):
    """
    The decoder takes in x_values, that is the target points and combines
    them with the represenation of the context points by concatenation.
    The resulting tensor is passed to an MLP that
    is asked to ouput the parameters for the sought after distribution, in this
    case a normal distribution. Thus we are looking for two parameters.
    The MLP returns two tensor obejects which hold a mean/ variance for each
    point y. Thus the shape of this output is batch_size,y_values,y_dim, 2

    Note the targets consist
    of both the context points as well as the target points,
    since the context points are a subset of the target points.


     Parameters
     ----------
     dimx: int: Dimension of each x value

     dimr: int: Dimension of each of the representations

     num_layers: int: Dimension of hidden layers

     num_neurons: int: Number of Neurons for each hidden layer
    """

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0,
                 activation='nn.ReLU()', batch_norm=False):
        super().__init__(insize, num_layers, num_neurons, dimout, dropout,
                         activation, batch_norm)

    def forward(self, x_values, r_values):
        """Takes x and r values, combines them and passes them twice to MLP.
        Thus we have one run for mu and one run for sigma"""
        input_as_pairs = torch.cat((x_values, r_values), dim=1)

        return self._process_input(input_as_pairs)


class BasicMLPAggregator(BasicMLP):
    """
    Goal of the this class is to learn the weights of as weighted average
    The input is the embedding from the encoder. For batch_size one,
     i.e. one function and x context points we need to learn x weights.
     That means that the input is num_context_points X dim_embedding
     In the simplest case this is multiplied by a dim_embedding X 1 weight
     vector to produce a num_context_points X 1 vector which is the weight
     vector
     That means that the learned weight vector has the dimensions
     x times 1. We transpose this an multiply it with the embedding tensor
    thus (batch_size, 1, x) *(batchsize_x, dim_embedding) =
    (batch_size, 1, dim_embedding)

    A theoretical derivation of this approach can be found here:
    https://arxiv.org/pdf/1802.04712.pdf

    """

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0,
                 activation='nn.ReLU()', batch_norm=False):
        """

        Parameters
        ----------

        insize : int
        The dimensions of the embedding
        num_layers : int
        num_neurons : int
        dimout : int
        dropout : int

        """
        super().__init__(insize, num_layers, num_neurons, dimout, dropout,
                         activation, batch_norm)

        self.softmax = nn.Softmax(dim=1)

    def aggregate(self, embedding, weights_for_average, batch_size,
                  normalize):
        weights_for_average = torch.transpose(weights_for_average, 1, 0)

        weights_for_average_batch = weights_for_average.view(
            batch_size, 1, -1)

        if normalize:
            normalized_weights = self.softmax(weights_for_average_batch)
        aggregation = torch.bmm(normalized_weights, embedding)

        return aggregation

    def forward(self, embedding, normalize=True):
        """

        Parameters
        ----------
        normalize
        embedding : tensor (batch_size, num_contxt, dim_embedding). As with
        all aggregation tasks the embedding is not stacked but expected in the
        batch view.

        Returns
        -------
        aggregation : tensor (batch_size, 1, dim_embedding) or the weighted
        average of the context points of the embedding for each batch.
        """
        batch_size, n_features, _ = embedding.size()
        stacked_embedding = embedding.reshape(
            batch_size * n_features, -1)
        weights_for_average = self._process_input(stacked_embedding)

        return self.aggregate(embedding, weights_for_average,
                              batch_size, normalize)


class GatedMLPAggregator(nn.Module):
    def __init__(self,
                 insize,
                 dimout,
                 # num_layers=2,
                 num_neurons=128,
                 # dropout=0,
                 # activation='nn.ReLU()',
                 # batch_norm=False
                 ):
        super().__init__()

        # self.attention = BasicMLP(
        #     insize=insize,
        #     dimout=num_neurons,
        #     num_layers=num_layers,
        #     num_neurons=num_neurons,
        #     dropout=dropout,
        #     activation=activation,
        #     batch_norm=batch_norm
        # )

        # self.gate = BasicMLP(
        #     insize=insize,
        #     dimout=num_neurons,
        #     num_layers=num_layers,
        #     num_neurons=num_neurons,
        #     dropout=dropout,
        #     activation=activation,
        #     batch_norm=batch_norm
        # )

        # The naming convention follows the original paper
        # Dauphin: https://arxiv.org/abs/1612.08083
        # and the at:
        # https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py

        self.attention_W = nn.Linear(insize, num_neurons)
        self.attenion_V = nn.Linear(insize, num_neurons)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(num_neurons, dimout)
        self.softmax = nn.Softmax(dim=1)

    def aggregate(self, embedding, weights_for_average, batch_size,
                  normalize):
        weights_for_average = torch.transpose(weights_for_average, 1, 0)
        weights_for_average_batch = weights_for_average.reshape(
            batch_size, 1, -1)

        if normalize:
            normalized_weights = self.softmax(weights_for_average_batch)

        aggregation = torch.bmm(normalized_weights, embedding)

        return aggregation

    def forward(self, embedding, normalize=True):
        batch_size, n_features, _ = embedding.size()
        stacked_embedding = embedding.reshape(
            batch_size * n_features, -1)
        # these two networks are both of size dim_embeddings X dim_hidden
        # attention_weights = self.attention.process_input(stacked_embedding)

        attention_weights_V = self.attenion_V(stacked_embedding)
        attention_weights_V = self.sigmoid(attention_weights_V)

        attention_weights_W = self.attention_W(stacked_embedding)
        attention_weights_W = self.tanh(attention_weights_W)

        # elementwise combination of the two weight vectors
        weights_for_average = self.fc(
            attention_weights_V * attention_weights_W)

        aggregation = self.aggregate(embedding, weights_for_average,
                                     batch_size, normalize)

        return aggregation


class TargetBasedAggregation:
    def __init__(self, distance_metric):
        method_name = distance_metric
        package_name = 'cnp.distance_metrics'
        package = import_module(package_name)
        self._distance_metric = getattr(package, method_name)

    def apply_target_based_attention(self, context_x, target_x, encoding,
                                     normalize=True):
        attention_weights = self._distance_metric(context_x, target_x)
        if normalize:
            attention_weights = self.softmax(attention_weights)

        target_based_encoding = torch.bmm(attention_weights, encoding)
        return target_based_encoding


class SimpleAggregator:

    def __init__(self, aggregation_operation):
        self.aggregation_type = aggregation_operation

        self.softmax = nn.Softmax(dim=1)

    def simple_aggregation(self, encoding):
        if self.aggregation_type == 'mean':
            aggregated_encoding = encoding.mean(1)
        elif self.aggregation_type == 'max':
            aggregated_encoding = encoding.max(1)[0]
        elif self.aggregation_type == 'sum':
            aggregated_encoding = encoding.sum(1)

        aggregated_encoding = aggregated_encoding.unsqueeze(1)

        return aggregated_encoding

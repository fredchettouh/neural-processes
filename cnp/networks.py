import torch
from torch import nn


def create_linear_layer(layer_specs, index, dropout=0):
    """
    Parameters
    ----------
    index: int: Indicates at which layer in the layer architecture
    specification the model currently is

    layer_specs: list: Holds the specification for all layers in the
        architecture

    dropout: float: specifies the dropout probability to be used in the dropout
        layer

    returns a list of length one with the layer of the network specified
    """
    lin_layer = nn.Linear(layer_specs[index], layer_specs[index + 1])
    relu_layer = nn.ReLU()
    dropout_layer = nn.Dropout(p=dropout)

    if dropout:
        return [lin_layer, relu_layer, dropout_layer]
    else:
        return [lin_layer, relu_layer]


def simple_aggregation(encoding, aggregation_operation):
    if aggregation_operation == 'mean':
        aggregated_encoding = encoding.mean(1)
    elif aggregation_operation == 'max':
        aggregated_encoding = encoding.max(1)
    elif aggregation_operation == 'sum':
        aggregated_encoding = encoding.sum(1)

    aggregated_encoding = aggregated_encoding.unsqueeze(1)

    return aggregated_encoding


class BasicMLP(nn.Module):

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0):
        super().__init__()

        # todo add dropout to first layer and add batch normalization

        self._insize = insize
        self._num_layers = num_layers
        self._num_neurons = num_neurons
        self._dimout = dimout
        self._dropout = dropout

        self._hidden_layers = [num_neurons for _ in range(num_layers)]

        _first_layer = [
            nn.Linear(self._insize, self._hidden_layers[0]),
            nn.ReLU()]

        if dropout:
            _first_layer.append(nn.Dropout(p=dropout))

        _hidden_layers = [
            create_linear_layer(self._hidden_layers, i, dropout)
            for i in range(len(self._hidden_layers) - 2)]
        _hidden_layers_flat = [
            element for inner in _hidden_layers for element in inner]

        _last_layer = [
            nn.Linear(self._hidden_layers[-1], self._dimout)]

        self._layers = _first_layer + _hidden_layers_flat + _last_layer

        self._process_input = nn.Sequential(*self._layers)


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

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0):
        super().__init__(insize, num_layers, num_neurons, dimout, dropout)

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

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0):
        super().__init__(insize, num_layers, num_neurons, dimout, dropout)

    def forward(self, x_values, r_values):
        """Takes x and r values, combines them and passes them twice to MLP.
        Thus we have one run for mu and one run for sigma"""
        input_as_pairs = torch.cat((x_values, r_values), dim=1)

        return self._process_input(input_as_pairs)


class BasicMLPAggregator(BasicMLP):
    """
    Goal of the this class is to learn the weights of an weighted average
    The input is the embedding from the encoder. For batch_size one,
     i.e. one function and x context points we need to learn x weights.
     That means that the learned weight vector has the dimensions
     x times 1. We transpose this an multiply it with the embedding tensor
    thus (batch_size, 1, x) *(batchsize_x, dim_embedding) =
     (batch_size, 1, dim_embedding)
    """

    def __init__(self, insize, num_layers, num_neurons, dimout, dropout=0):
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
        super().__init__(insize, num_layers, num_neurons, dimout, dropout)

    @staticmethod
    def aggregate(embedding, weights_for_average, batch_size, normalize=False):
        weights_for_average = torch.transpose(weights_for_average, 1, 0)
        stacked_weights_for_average = weights_for_average.view(
            batch_size, 1, -1)
        aggregation = torch.bmm(stacked_weights_for_average, embedding)
        if normalize:
            aggregation = aggregation / aggregation.sum()

        return aggregation

    def forward(self, embedding):
        """

        Parameters
        ----------
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
                              batch_size)


class BaseAggregator(nn.Module):

    def __init__(
            self, insize, num_layers, num_neurons, dimout=1, padding=False):
        super().__init__()
        self._insize = insize
        self._dimout = dimout
        self._num_layers = num_layers
        self._num_neurons = num_neurons
        self._padding = padding

    def _zero_padding(self, embedding):

        if self._zero_padding:
            batch_size, n_observations, n_features = embedding.size()
            zero_target = torch.zeros(batch_size, n_observations, self._insize)
            zero_target[:, :, :n_features] = embedding
            return zero_target
        else:
            return embedding

    def init_hidden(self, num_tensors, batch_size):
        base = next(self.parameters()).data
        if num_tensors == 1:
            hidden = base.new(
                self._num_layers, batch_size, self._num_neurons).zero_()
        else:
            hidden = (
                base.new(
                    self._num_layers, batch_size, self._num_neurons).zero_(),
                base.new(
                    self._num_layers, batch_size, self._num_neurons).zero_())

        return hidden


class MLPAggregator(BaseAggregator):
    def __init__(
            self,
            insize,
            num_layers: int = 3,
            num_neurons: int = 128,
            dimout=1,
            dropout: float = 0,
            padding: bool = False
    ):
        """

        Parameters
        ----------
        maxcontxt :
        dimout :
        num_layers :
        num_neurons :
        dropout :
        """
        super().__init__(insize, num_layers, num_neurons, dimout, padding)

        self._hidden_layers = [num_neurons for _ in range(num_layers)]

        _first_layer = [
            nn.Linear(self._insize, self._hidden_layers[0]),
            nn.ReLU()]
        _hidden_layers = [
            create_linear_layer(self._hidden_layers, i, dropout)
            for i in range(len(self._hidden_layers) - 2)]
        _hidden_layers_flat = [
            element for inner in _hidden_layers for element in inner]

        _last_layer = [
            nn.Linear(self._hidden_layers[-1], self._dimout)]

        self._layers = _first_layer + _hidden_layers_flat + _last_layer

        self._process_input = nn.Sequential(*self._layers)

    def forward(self, embedding, hidden):
        """
        Parameters
        ----------
        hidden
        embedding: torch.Tensor: Shape (batch_size*num_contxt, dimr)

        """
        padded_embedding = self._zero_padding(embedding)
        batch_size, n_features, _ = padded_embedding.size()
        padded_embedding_stacked = padded_embedding.view(
            batch_size * n_features, -1)

        output = self._process_input(padded_embedding_stacked)
        # we return None to pass to to hidden
        _ = None

        return output, _


class AggrRNN(BaseAggregator):
    # TODO is there a change if we feed a different shape to the
    # network i.e. reshaping from 64,128,9 t0 128,64,9-->or is this the same
    # TODO do we want to keep the hidden state alive across epochs?
    def __init__(self, insize, num_layers, num_neurons,
                 dimout, batch_first=True):
        super().__init__(insize, num_layers, num_neurons, dimout)

        self.rnn = nn.RNN(insize, num_neurons, num_layers,
                          batch_first=batch_first)

        self.fc = nn.Linear(num_neurons, dimout)

    # todo this can be taken out i think because with h_0=none it will
    # initialize to zero
    # def init_h0(self, batch_size):
    #     h_0 = torch.zeros(self._num_layers, batch_size, self._num_neurons)
    #     return h_0

    def forward(self, encoding, hidden=None):
        # encoding comes in shape batch_size, num_obs, num_features
        batch_size, seq_len, _ = encoding.size()
        padded_encoding = self._zero_padding(encoding)
        if not hidden:
            hidden = self.init_hidden(num_tensors=1, batch_size=batch_size)

        # encoding is passed due to batch_size=first
        output, hidden = self.rnn(padded_encoding, hidden)
        # output is reshaped to fit in the linear layers
        # Todo find out why reshape works but view does not

        output = output.reshape(batch_size * seq_len, -1)
        output = self.fc(output)

        return output, hidden


class AggrLSTM(BaseAggregator):

    def __init__(self, insize, num_layers, num_neurons,
                 dimout, batch_first):
        super().__init__(insize, num_layers, num_neurons, dimout)

        self.lstm = nn.LSTM(insize, num_neurons, num_layers,
                            batch_first=batch_first)

        self.fc = nn.Linear(num_neurons, dimout)

    def forward(self, encoding, hidden=None):
        batch_size, seq_len, _ = encoding.size()
        padded_encoding = self._zero_padding(encoding)

        if not hidden:
            hidden = self.init_hidden(num_tensors=2, batch_size=batch_size)

        output, hidden = self.lstm(padded_encoding, hidden)
        # output is reshaped to fit in the linear layers
        # Todo find out why reshape works but view does not

        output = output.reshape(batch_size * seq_len, -1)
        output = self.fc(output)

        return output, hidden


if __name__ == "__main__":
    encoder = Encoder(dimx=1, dimy=1, dimr=128,
                      num_layers=4, num_neurons=128, dropout=0)
    decoder = Decoder(dimx=1, dimr=128, dimparam=2, num_layers=3,
                      num_neurons=128, dropout=0.2)
    print(encoder, decoder)

import torch
from torch import nn


def create_linear_layer(layer_specs, index, dropout=0):
    """
    Parameters
    ----------
    index: int: Indicates at which layer in the layer architecture specification
    the model currently is

    layer_specs: list: Holds the specification for all layers in the architecture

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


class Encoder(nn.Module):
    """This class maps each x_i, y_i context point to a representation r_i
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

    def __init__(self, dimx: int, dimy: int, dimr: int,
                 num_layers: int, num_neurons: int,
                 dropout: float = 0) -> None:
        super().__init__()

        self._dimx = dimx
        self._dimy = dimy
        self._dimr = dimr
        self._hidden_layers = [num_neurons for _ in range(num_layers)]

        _first_layer = [
            nn.Linear(self._dimx + self._dimy, self._hidden_layers[0]),
            nn.ReLU()]

        _hidden_layers = [
            create_linear_layer(self._hidden_layers, i, dropout)
            for i in range(len(self._hidden_layers) - 2)]
        _hidden_layers_flat = [
            element for inner in _hidden_layers for element in inner]

        _last_layer = [
            nn.Linear(self._hidden_layers[-2], self._hidden_layers[-1])]

        self._layers = _first_layer + _hidden_layers_flat + _last_layer

        self._process_input = nn.Sequential(*self._layers)

    def forward(self, x_values, y_values):
        """
        Parameters
        ----------
        x_values: torch.Tensor: Shape (batch_size, dimx)

        y_values: torch.Tensor: Shape (batch_size, dimy)
        """
        input_as_pairs = torch.cat((x_values, y_values), dim=1)

        return self._process_input(input_as_pairs)


class Decoder(nn.Module):
    """The decoder takes in x_values, that is the target points and combines
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
    def __init__(self, dimx, dimr, dimparam, num_layers, num_neurons, dropout=0):
        super().__init__()

        self._dimx = dimx
        self._dimr = dimr
        self._dimparam = dimparam
        self._hidden_layers = [num_neurons for _ in range(num_layers)]

        _first_layer = [
            nn.Linear(self._dimx + self._dimr, self._hidden_layers[0]),
            nn.ReLU()]
        if dropout:
            _first_layer.append(nn.Dropout(p=dropout))

        _hidden_layers = [
            create_linear_layer(self._hidden_layers,i , dropout)
            for i in range(len(self._hidden_layers) - 2)]
        _hidden_layers_flat = [
            element for inner in _hidden_layers for element in inner]
        _last_layer = [nn.Linear(self._hidden_layers[-1], self._dimparam)]

        self._layers = _first_layer + _hidden_layers_flat + _last_layer

        self._process_input = nn.Sequential(*self._layers)

    def forward(self, x_values, r_values):
        """Takes x and r values, combines them and passes them twice to MLP.
        Thus we have one run for mu and one run for sigma"""

        input_as_pairs = torch.cat((x_values, r_values), dim=1)

        return self._process_input(input_as_pairs)


if __name__ == "__main__":
    encoder = Encoder(dimx=1, dimy=1, dimr=128,
                      num_layers=4, num_neurons=128,dropout=0)
    decoder = Decoder(dimx=1, dimr=128, dimparam=2, num_layers=3,
                      num_neurons=128, dropout=0.2)
    print(encoder, decoder)


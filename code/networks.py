import torch
from torch import nn
import numpy as np


class Encoder(nn.Module):
    """This class maps each x_i, y_i context point to a representation r_i
    To learn this Representation we are using a Multi Layer Perceptron
    The input shape will be batch_size, num_context_points, x_dim

    The input to the encoder are the value pairs, thus the dimensions are
    Batch_Size, (dimx+dimy). The Pytorch automatically pases the values sequentially
    through the ANN.
    The last layer will not have an activation function because we want the pure represenation.

    Parameters
    ----------

    dimx : int
        Dimesion of each x value

    dimy : int
        Dimesion of each y value

    dimr : int
        Dimension of output representation

    num_layers : int
        Dimension of hidden layers

    num_neurons: int

        Number of Neurons for each hidden layer

    """

    def __init__(self, dimx, dimy, dimr, num_layers, num_neurons):
        super().__init__()

        self._dimx = dimx
        self._dimy = dimy
        self._dimr = dimr
        self._hidden_layers = [num_neurons for _ in range(num_layers)]

        _first_layer = [nn.Linear(self._dimx + self._dimy, self._hidden_layers[0]), nn.ReLU()]

        _hidden_layers = list(np.array([
            [nn.Linear(self._hidden_layers[_], self._hidden_layers[_ + 1]), nn.ReLU()]
            for _ in range(len(self._hidden_layers) - 2)
        ]).flatten())

        _last_layer = [nn.Linear(self._hidden_layers[-2], self._hidden_layers[-1])]

        self._layers = _first_layer + _hidden_layers + _last_layer

        self._process_input = nn.Sequential(*self._layers)

    def forward(self, x_values, y_values):
        """
        Takes the context points x and y,
        concatenates them into value pairs
        and passes them through the MLP

        Parameters
        ----------

        x_values : torch.Tensor
            Shape (batch_size, dimx)

        y_values : torch.Tensor
            Shape (batch_size, dimy)

        """

        input_as_pairs = torch.cat((x_values, y_values), dim=1)

        return self._process_input(input_as_pairs)


class Decoder(nn.Module):
    """The decoder takes in x_values, that is the target points and combines them with
    the represenation of the context points by concatenation. The resulting tensor is passed to an MLP that
    is asked to ouput the parameters for the sought after distribution, in this case
    a normal distribution. Thus we are looking for two parameters. The MLP returns two tensor obejects
    which hold a mean/ variance for each point y. Thus the shape of this output is
    batch_size,y_values,y_dim, 2

    Note the targets consist
    of both the context points as well as the target points, since the context points
    are a subset of the target points.


    Parameters
    ----------

    dimx : int
        Dimension of each x value

    dimr : int
        Dimension of each of the representations

    num_layers : int
        Dimension of hidden layers

    num_neurons: int

        Number of Neurons for each hidden layer

    """

    def __init__(self, dimx, dimr, dimparam, num_layers, num_neurons):
        super().__init__()

        self._dimx = dimx
        self._dimr = dimr
        self._dimparam = dimparam
        self._hidden_layers = [num_neurons for _ in range(num_layers)]

        _first_layer = [nn.Linear(self._dimx + self._dimr, self._hidden_layers[0]), nn.ReLU()]

        _hidden_layers = list(np.array([
            [nn.Linear(self._hidden_layers[_], self._hidden_layers[_ + 1]), nn.ReLU()]
            for _ in range(len(self._hidden_layers) - 1)
        ]).flatten())

        _last_layer = [nn.Linear(self._hidden_layers[-1], self._dimparam)]

        self._layers = _first_layer + _hidden_layers + _last_layer

        self._process_input = nn.Sequential(*self._layers)

    def forward(self, x_values, r_values):
        """Takes x and r values, combines them and passes them twice to MLP.
        Thus we have one run for mu and one run for sigma"""    

        input_as_pairs = torch.cat((x_values, r_values), dim=1)

        return self._process_input(input_as_pairs)
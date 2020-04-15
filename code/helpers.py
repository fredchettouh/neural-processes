import torch
from torch.nn.functional import softplus


class Helper():

    def __init__(self):
        pass

    @staticmethod
    def scale_shift_uniform(a=0, b=1, *size):
        return torch.rand(size=(size)) * (a - b) + b

    @staticmethod
    def list_np_to_sensor(list_of_arrays, stack=True):
        if stack:
            return torch.stack([array for array in list_of_arrays])
        else:
            [array for array in list_of_arrays]

    @staticmethod
    def transform_var(var_tensor):

        """This function takes a learned variance tensor and transforms
        it following the methodology in Empirical Evaluation of Neural Process Objectives.
        This ensures that the covariance matrix is positive definite and a multivariate
        Gaussian can be constructed.
        Next it pads the diagonal with zeroes to create a covariance matrix for sampling.
       """
        transformed_variance = 0.1 + 0.9 * softplus(var_tensor)
        cov_matrix = torch.diag_embed(transformed_variance)
        return cov_matrix

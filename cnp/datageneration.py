#!/usr/bin/env python


## Todo: rewrite kernel so that it works for higher order tensors
# Todo: add seeds to make experiments comparable


import torch
import argparse
import numpy as np
from .helpers import Helper
from copy import copy


class DataGenerator:
    """Takes a specific dimensions and a kernel argument and returns a dataset 
    generated using the Kernel
    Parameters:
    -----------
    xdim : int: Dimension of each x value
        
    ydim : int: Dimension of each y value
        
    range_x : tuple: min, max values of range on which to define the data
    
    num_instances : int : Number data instances to generate

    steps: tuple: number of x values to create
    """

    def __init__(self, xdim=1, range_x=(None, None), steps=400):
        self._xdim = xdim
        self._xmin = range_x[0]
        self._xmax = range_x[1]
        self._steps = steps

    def _create_shuffled_linspace(self):
        lspace = torch.linspace(self._xmin, self._xmax, self._steps)[:, None]
        temp_lspace = copy(lspace)
        for i in range(1, self._xdim):
            perm = torch.randperm(self._steps)
            lspace_shuffled = temp_lspace[perm]
            lspace = torch.cat((lspace, lspace_shuffled), dim=1)

        return lspace

    def generate_curves(self):
        raise Exception('Not implemented as base level')

    def generate_loader_on_fly(self, batch_size, kwargs, purpose):
        kwargs['purpose'] = purpose

        X, y = self.generate_curves(**kwargs)

        loader = Helper.create_loader(
            X, y, batch_size)

        return loader

    @staticmethod
    def generate_from_single_function(X, y, batch_size, shuffle=True):
        if shuffle:
            shuffled_tensors = Helper.shuffletensor(X, y)
            X, y = shuffled_tensors[0], shuffled_tensors[1]

        loader = Helper.create_loader(
            X, y, batch_size)
        return loader


class GaussianProcess(DataGenerator):
    pass

    def _rbf_kernel(self, length_scale, gamma, x):
        y = x
        # getting x1^2+x2^2....xp^2 for each observation
        xrow = torch.sum(x * x, 1)
        yrow = torch.sum(y * y, 1)
        # reshaping x and y sums so that we can add each y^2 row sum
        # to each element of the vector xrow
        # thus creating a matrix of pairwise sums x^2+y^2
        x_res = xrow.reshape(x.shape[0], 1)
        y_res = yrow.reshape(1, x.shape[0])
        # this creats the xy product
        xy = torch.mm(x, y.t())
        # adding everything together of the form x^2+y^2-2xy scaling it 
        kernel = torch.exp(-gamma * ((x_res + y_res - 2 * xy) / length_scale))
        return kernel

    def generate_curves(self, noise=1e-4, length_scale=0.4, gamma=1,
                        num_instances_train=None, num_instances_vali=None,
                        num_instances_test=None, purpose=None):
        x_values = self._create_shuffled_linspace()
        kernel = self._rbf_kernel(length_scale, gamma, x_values)
        kernel = kernel + torch.eye(self._steps) * noise
        cholesky_decomp = torch.cholesky(kernel)
        datasets = []
        if purpose == 'train':
            num_instances = num_instances_train
        elif purpose == 'vali':
            num_instances = num_instances_vali
        elif purpose == 'test':
            num_instances = num_instances_test

        for _ in range(num_instances):
            # creating as many standard
            standard_normals = torch.normal(0, 1, (self._steps, 1))
            func_x = cholesky_decomp @ standard_normals
            datasets.append(func_x)

        datasets = Helper.list_np_to_tensor(datasets)
        x_values = x_values.repeat(datasets.shape[0], 1, 1)

        return x_values, datasets


class PolynomialRegression(DataGenerator):
    pass

    def generate_curves(
            self,
            mu_gen=0,
            sigma_gen=1,
            mu_noise=0,
            sigma_noise=0.04,
            min_coef=-1,
            max_coef=1,
            num_instances_train=None,
            num_instances_vali=None,
            num_instances_test=None,
            purpose=None,
            seed=None):

        if purpose == 'train':
            num_instances = num_instances_train
        elif purpose == 'vali':
            num_instances = num_instances_vali
        elif purpose == 'test':
            num_instances = num_instances_test

        x_values = torch.normal(
            mean=mu_gen,
            std=sigma_gen,
            size=(num_instances, self._steps, self._xdim),
            dtype=torch.float64)
        x_values_copy = copy(x_values)

        for i in range(0, self._xdim):
            x_values_copy = torch.cat((
                x_values_copy,
                (x_values_copy[:, :, i] ** 2)[:, :, None],
                (x_values_copy[:, :, i] ** 3)[:, :, None]),
                dim=2)
        coefs = Helper.scale_shift_uniform(min_coef, max_coef,
                                           *(num_instances, self._xdim * 3, 1))

        noise = torch.normal(
            mean=mu_noise, std=sigma_noise, size=(num_instances, 1, 1),
            dtype=torch.float64)

        func_values = torch.bmm(x_values_copy, coefs) + noise

        return x_values.float(), func_values.float()



# class TwoDImageRegression:
#     def __init__(self, width, height):
#
#
#

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-num_instances", required=True,
                    help="Number of curves to generate")
    ap.add_argument("-noise", required=True,
                    help="Noise to add to diagonal to make the Cholesky work")
    ap.add_argument("-length_scale", required=True, help="Scaling parameter")
    ap.add_argument("-gamma", required=True, help="Scaling parameter")

    args = vars(ap.parse_args())
    gen = DataGenerator()
    x_values, data = gen.generate_curves(args['num_instances'], args['noise'],
                                         args['length_scale'], args['gamma'])

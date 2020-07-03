#!/usr/bin/env python

import torch
import numpy as np
from .helpers import Helper
from copy import copy
from torchvision import datasets, transforms
from itertools import product
from sklearn.metrics import pairwise


class DataGenerator:
    pass

    def _create_shuffled_linspace(self):
        lspace = torch.linspace(self._xmin, self._xmax, self._steps)[:, None]
        temp_lspace = copy(lspace)
        for i in range(1, self._xdim):
            perm = torch.randperm(self._steps)
            lspace_shuffled = temp_lspace[perm]
            lspace = torch.cat((lspace, lspace_shuffled), dim=1)

        return lspace

    def create_mnist_coordinates(self, width, height):
        x_1 = np.arange(0, width)
        x_2 = np.arange(0, height)
        x_values = torch.Tensor(list(product(x_1, x_2)))
        x_values = x_values[None, :, :]
        return x_values

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
    def __init__(self, xdim=1, range_x=(None, None), steps=400):
        self._xdim = xdim
        self._xmin = range_x[0]
        self._xmax = range_x[1]
        self._steps = steps

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

        datasets = []
        if purpose == 'train':
            num_instances = num_instances_train
            x_values = Helper.scale_shift_uniform(
                self._xmin, self._xmax, *(self._steps, self._xdim)).float()

        elif purpose == 'vali':
            num_instances = num_instances_vali
            x_values = self._create_shuffled_linspace()

        elif purpose == 'test':
            num_instances = num_instances_test
            x_values = self._create_shuffled_linspace()

        kernel = self._rbf_kernel(length_scale, gamma, x_values)
        kernel = kernel + torch.eye(self._steps) * noise
        cholesky_decomp = torch.cholesky(kernel)

        for _ in range(num_instances):
            # creating as many standard
            standard_normals = torch.normal(0, 1, (self._steps, 1))
            func_x = cholesky_decomp @ standard_normals
            datasets.append(func_x)

        datasets = Helper.list_np_to_tensor(datasets)
        x_values = x_values.repeat(datasets.shape[0], 1, 1)

        return x_values, datasets


class PolynomialRegression(DataGenerator):

    def __init__(self, xdim=1, range_x=(None, None), steps=400):
        self._xdim = xdim
        self._xmin = range_x[0]
        self._xmax = range_x[1]
        self._steps = steps

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


class TwoDImageRegression(DataGenerator):

    def __init__(self,
                 width,
                 height,
                 scale_mean=0.5,
                 scale_std=0.5,
                 link='~/.pytorch/MNIST_data/',
                 share_train_data=0.8):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((scale_mean,), (scale_std,))
            ])

        traindata = datasets.MNIST(
            link, download=True, train=True, transform=transform).data

        self._testset = datasets.MNIST(
            link, download=True, train=False, transform=transform).data

        idx_train = int(len(traindata) * share_train_data)
        self._trainset = traindata[0: idx_train]
        self._valiset = traindata[idx_train:]

        self._width = width
        self._height = height

    @staticmethod
    def select_mnist_samples(tensor_list, num_instances, width, height):

        max_idx = len(tensor_list)
        idx = torch.randperm(max_idx)[:num_instances]
        func_values = tensor_list[idx]
        func_values = func_values.view(num_instances, width * height)
        func_values = func_values[:, :, None] / 255.0
        return func_values

    def generate_curves(
            self,
            num_instances_train=None,
            num_instances_vali=None,
            num_instances_test=None,
            purpose=None,
            seed=None):

        if purpose == 'train':
            num_instances = num_instances_train
            tensor_list = self._trainset
        elif purpose == 'vali':
            num_instances = num_instances_vali
            tensor_list = self._valiset
        elif purpose == 'test':
            num_instances = num_instances_test
            tensor_list = self._testset

        x_values = self.create_mnist_coordinates(self._width, self._height)
        x_values = x_values.repeat(num_instances, 1, 1)

        func_values = self.select_mnist_samples(
            tensor_list, num_instances, self._width, self._height)

        return x_values.float(), func_values.float()


class PairwiseKernel(DataGenerator):
    def __init__(self, xdim, range_x, steps):
        self._xdim = xdim
        self._xmin = range_x[0]
        self._xmax = range_x[1]
        self._steps = steps

    def generate_curves(
            self,
            kernel_name,
            noise,
            num_instances_train=None,
            num_instances_vali=None,
            num_instances_test=None,
            purpose=None, **kwargs):

        x_values = self._create_shuffled_linspace()
        kernel = torch.tensor(
            pairwise.pairwise_kernels(
                x_values, x_values,
                kernel_name, kwargs),
            dtype=torch.float64)

        kernel = kernel + torch.eye(self._steps) * noise
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
            func_x = kernel.float() @ standard_normals.float()
            datasets.append(func_x)

        datasets = Helper.list_np_to_tensor(datasets)
        x_values = x_values.repeat(datasets.shape[0], 1, 1)

        return x_values, datasets

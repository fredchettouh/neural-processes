# pytorch imports
import torch
from torch import nn
from torch import optim
# from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


import numpy as np
import matplotlib.pyplot as plt

# custom imports

from networks import Encoder, Decoder
from helpers import Helper


class Experiment(nn.Module):

    def __init__(self,
                 n_epochs,
                 lr,
                 max_funcs,
                 max_contx,
                 min_contx,
                 dim_observation,
                 dimx=1,
                 dimy=1,
                 dimr=50,
                 dimout=2,
                 dim_encoder=[128, 128, 128],
                 dim_decoder=[128, 128, 128],
                 train_on_gpu=False,
                 print_after=100):
        super().__init__()

        self._n_epochs = n_epochs
        self._lr = lr
        self._max_trgts = max_funcs
        self._max_contx = max_contx
        self._min_contx = min_contx
        self._dim_observation = dim_observation
        self._dimx = dimx
        self._dimy = dimy
        self._dimr = dimr
        self._dimout = dimout
        self._dim_encoder = dim_encoder
        self._dim_decoder = dim_decoder
        self._train_on_gpu = train_on_gpu
        self._print_after = print_after

        self._encoder = Encoder(self._dimx, self._dimy, self._dimr, self._dim_encoder)
        self._decoder = Decoder(self._dimx, self._dim_encoder[-1], self._dimout, self._dim_decoder)

    def _prep_data(self, xvalues, funcvalues, training=True):

        if self._train_on_gpu:
            xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

        if training:
            contxt_idx = self._get_sample_indexes()
            target_y = funcvalues[:, func_idx, :]
            target_x = xvalues[:, func_idx, :]
        else:
            contxt_idx = self._get_sample_indexes(both=False)
            target_y = funcvalues[:, :, :]
            target_x = xvalues[:, :, :]

        batch_size = xvalues.shape[0]

        num_contxt, num_trgt = len(contxt_idx), self._dim_observation

        context_y = funcvalues[:, contxt_idx, :]
        context_x = xvalues[:, contxt_idx, :]

        # the encoding is stacked to ensure a one dimensional input
        context_y_stacked = context_y.view(batch_size * num_contxt, -1)
        context_x_stacked = context_x.view(batch_size * num_contxt, -1)

        # running the context values through the encoding
        encoding = self._encoder(context_x_stacked, context_y_stacked)
        encoding = encoding.view(batch_size, num_contxt, -1)
        # averaging the encoding
        encoding_avg = encoding.mean(1)
        # we need to unsqueeze and repeat the embedding
        # because we need to pair it with every target
        encoding_avg = encoding_avg.unsqueeze(1)
        encoding_exp = encoding_avg.repeat(1, num_trgt, 1)

        encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)
        target_x_stacked = target_x.view(batch_size * num_trgt, -1)

        decoding = self._decoder(target_x_stacked, encoding_stacked)
        decoding_rshp = decoding.view(batch_size, num_trgt, -1)

        mu, sigma = decoding_rshp[:, :, 0].unsqueeze(-1), decoding_rshp[:, :, 1].unsqueeze(-1)
        sigma_transformed = Helper.transform_var(sigma)
        distribution = Normal(loc=mu, scale=sigma_transformed)

        return batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution

    def _get_sample_indexes(self, both=True):
        num_contxt = np.random.randint(self._min_contx, self._max_contx)
        num_trgts = np.random.randint(self._max_contx, self._max_trgts)
        trgts_idx = np.random.choice(np.arange(0, self._dim_observation), num_trgts)
        contxt_idx = trgts_idx[:num_contxt]
        if both:
            return trgts_idx, contxt_idx
        else:
            return contxt_idx


    def plot_run(self, batch_size, contxt_idx, xvalues, funcvalues, target_y, target_x, mu, cov_matrix):

        random_function = np.random.randint(0, batch_size)
        context_y_plot = funcvalues[random_function, contxt_idx, :].flatten()
        context_x_plot = xvalues[random_function, contxt_idx, :].flatten()
        y_plot = target_y[random_function, :, :].flatten()
        x_plot = target_x[random_function, :, :].flatten()
        mu_plot = mu[random_function, :, :].flatten()
        var_plot = cov_matrix[random_function, :, :].flatten()
        plt.scatter(x_plot, y_plot, color='red')
        plt.scatter(context_x_plot, context_y_plot, color='black')
        plt.plot(x_plot, mu_plot, color='blue')
        plt.fill_between(x_plot, y1=mu_plot + var_plot, y2=mu_plot - var_plot, alpha=0.2)
        plt.show()
        plt.close()

    def _validation_run(self, valiloader, current_epoch, plotting=True):

        self._encoder.eval()
        self._decoder.eval()

        running_vali_loss = 0
        self.eval()

        with torch.no_grad():

            for xvalues, funcvalues in valiloader:

                batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution = self._prep_data(xvalues, funcvalues, training=False)
                vali_loss = distribution.log_prob(target_y)
                vali_loss = -torch.mean(vali_loss)
                running_vali_loss += vali_loss.item()
            else:
                print(f' Validation loss after {current_epoch} equals {running_vali_loss / (len(valiloader))}')
                if plotting:
                    self.plot_run(batch_size, contxt_idx, xvalues, funcvalues, target_y, target_x, mu, sigma_transformed)

    def run_training(self, trainloader, valiloader=None, plotting=False):
        # defining the Encoder and the Decoder nstances


        self._encoder.train()
        self._decoder.train()


        optimizer = optim.Adam(self._decoder.parameters())
        mean_epoch_loss = []

        for epoch in range(self._n_epochs):
            running_loss = 0
            #         get sample indexes
            for xvalues, funcvalues in trainloader:
                # if self._train_on_gpu:
                    # xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                optimizer.zero_grad()
                # we sample for every single batch
                # Setting the batch size here is not ideal
                # batch_size = xvalues.shape[0]
                #
                # func_idx, contxt_idx = self._get_sample_indexes()
                # num_trgt, num_contxt = len(func_idx), len(contxt_idx)
                #
                # target_y = funcvalues[:, func_idx, :]
                # target_x = xvalues[:, func_idx, :]
                #
                # context_y = funcvalues[:, contxt_idx, :]
                # context_x = xvalues[:, contxt_idx, :]
                #
                # # the encoding is stacked to ensure a one dimensional input
                # context_y_stacked = context_y.view(batch_size * num_contxt, -1)
                # context_x_stacked = context_x.view(batch_size * num_contxt, -1)
                #
                # # running the context values through the encoding
                # encoding = self._encoder(context_x_stacked, context_y_stacked)
                # encoding = encoding.view(batch_size, num_contxt, -1)
                # # averaging the encoding
                # encoding_avg = encoding.mean(1)
                # # we need to unsqueeze and repeat the embedding
                # # because we need to pair it with every target
                # encoding_avg = encoding_avg.unsqueeze(1)
                # encoding_exp = encoding_avg.repeat(1, num_trgt, 1)
                #
                # encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)
                # target_x_stacked = target_x.view(batch_size * num_trgt, -1)
                #
                # decoding = self._decoder(target_x_stacked, encoding_stacked)
                # decoding_rshp = decoding.view(batch_size, num_trgt, -1)
                #
                # #
                # mu, sigma = decoding_rshp[:, :, 0].unsqueeze(-1), decoding_rshp[:, :, 1].unsqueeze(-1)
                #
                # cov_matrix = Helper.transform_var(sigma)
                # distribution = MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)
                batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution = self._prep_data(xvalues, funcvalues, training=False)
                loss = distribution.log_prob(target_y)
                loss = -torch.mean(loss)
                running_loss += loss

                loss.backward()
                optimizer.step()
            else:
                mean_epoch_loss.append(running_loss / len(trainloader))

                if epoch % self._print_after == 0 and epoch > 0:
                    print(f'Mean loss at epoch {epoch} : {mean_epoch_loss[-1]}')
                    # if self._train_on_gpu:
                        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
                    if valiloader:
                        self._validation_run(valiloader, epoch,plotting)
                        self._encoder.train(), self._decoder.train()

        return self._decoder.state_dict()

    def run_test(self, file_path_weights, testloader):
        running_mse = 0
        state_dict = torch.load(file_path_weights)
        self._decoder.load_state_dict(state_dict)

        self._encoder.eval()
        self._decoder.eval()

        with torch.no_grad():

            for xvalues, funcvalues in testloader:
                # if self._train_on_gpu:
                #     xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()
                #
                # batch_size = xvalues.shape[0]
                #
                # target_y = funcvalues[:, :, :]
                # target_x = xvalues[:, :, :]
                #
                # contxt_idx = self._get_sample_indexes(both=False)
                # num_contxt, num_trgt = len(contxt_idx), self._dim_observation
                #
                # context_y = funcvalues[:, contxt_idx, :]
                # context_x = xvalues[:, contxt_idx, :]
                #
                # # the encoding is stacked to ensure a one dimensional input
                # context_y_stacked = context_y.view(batch_size * num_contxt, -1)
                # context_x_stacked = context_x.view(batch_size * num_contxt, -1)
                #
                # # running the context values through the encoding
                # encoding = self._encoder(context_x_stacked, context_y_stacked)
                # encoding = encoding.view(batch_size, num_contxt, -1)
                # # averaging the encoding
                # encoding_avg = encoding.mean(1)
                # # we need to unsqueeze and repeat the embedding
                # # because we need to pair it with every target
                # encoding_avg = encoding_avg.unsqueeze(1)
                # encoding_exp = encoding_avg.repeat(1, num_trgt, 1)
                #
                # encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)
                # target_x_stacked = target_x.view(batch_size * num_trgt, -1)
                #
                # decoding = self._decoder(target_x_stacked, encoding_stacked)
                # decoding_rshp = decoding.view(batch_size, num_trgt, -1)

                # mu, sigma = decoding_rshp[:, :, 0].unsqueeze(-1), decoding_rshp[:, :, 1].unsqueeze(-1)
                # mu = mu.unsqueeze(-1)
                # cov_matrix = Helper.transform_var(sigma)
                batch_size, target_x, target_y, context_x, contxt_idx, context_y, mu, sigma_transformed, distribution = self._prep_data(xvalues, funcvalues, training=False)
                mse = ((mu-target_y)**2).mean(1).mean(0)
                running_mse += mse.item()

                self.plot_run(batch_size, contxt_idx, xvalues, funcvalues, target_y, target_x, mu, sigma_transformed)

            else:
                test_set_mse = running_mse/len(testloader)
                return test_set_mse



# pytorch imports
import torch
from torch import nn
from torch.nn.functional import softplus
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal


import numpy as np

# custom imports

from networks import Encoder, Decoder
from helpers import Helper


class Trainer(nn.Module):

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
        self.train_on_gpu = train_on_gpu
        self._print_after = print_after

    def _get_sample_indexes(self, both=True):
        num_contxt = np.random.randint(self._min_contx, self._max_contx)
        num_trgts = np.random.randint(self._max_contx, self._max_trgts)
        trgts_idx = np.random.choice(np.arange(0, self._dim_observation), num_trgts)
        contxt_idx = trgts_idx[:num_contxt]
        if both:
            return trgts_idx, contxt_idx
        else:
            return contxt_idx

    def _validation_run(self, valiloader, encoder, decoder, current_epoch):

        encoder.eval()
        encoder.eval()

        running_vali_loss = 0
        self.eval()

        with torch.no_grad():

            for xvalues, funcvalues in valiloader:
                if self.train_on_gpu:
                    xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                batch_size = xvalues.shape[0]

                target_y = funcvalues[:, :, :]
                target_x = xvalues[:, :, :]

                contxt_idx = self._get_sample_indexes(both=False)
                num_contxt, num_trgt = len(contxt_idx), self._dim_observation

                context_y = funcvalues[:, contxt_idx, :]
                context_x = xvalues[:, contxt_idx, :]

                # the encoding is stacked to ensure a one dimensional input
                context_y_stacked = context_y.view(batch_size * num_contxt, -1)
                context_x_stacked = context_x.view(batch_size * num_contxt, -1)

                # running the context values through the encoding
                encoding = encoder(context_x_stacked, context_y_stacked)
                encoding = encoding.view(batch_size, num_contxt, -1)
                # averaging the encoding
                encoding_avg = encoding.mean(1)
                # we need to unsqueeze and repeat the embedding
                # because we need to pair it with every target
                encoding_avg = encoding_avg.unsqueeze(1)
                encoding_exp = encoding_avg.repeat(1, num_trgt, 1)

                encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)
                target_x_stacked = target_x.view(batch_size * num_trgt, -1)

                decoding = decoder(target_x_stacked, encoding_stacked)
                decoding_rshp = decoding.view(batch_size, num_trgt, -1)

                mu, sigma = decoding_rshp[:, :, 0], decoding_rshp[:, :, 1]
                cov_matrix = Helper.transform_var(sigma)
                distribution = MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)

                vali_loss = distribution.log_prob(target_y.squeeze(-1))
                vali_loss = -torch.mean(vali_loss)
                running_vali_loss += vali_loss.item()
            else:
                print(f' Validation loss after {current_epoch} equals {running_vali_loss / (len(valiloader))}')

    def run_training(self, trainloader, valiloader=None):
        # defining the Encoder and the Decoder nstances
        encoder = Encoder(self._dimx, self._dimy, self._dimr, self._dim_encoder)
        decoder = Decoder(self._dimx, self._dim_encoder[-1], self._dimout, self._dim_decoder)

        encoder.train()
        decoder.train()

        if self.train_on_gpu:
            xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

        optimizer = optim.Adam(decoder.parameters())
        mean_epoch_loss = []

        for epoch in range(self._n_epochs):
            running_loss = 0
            #         get sample indexes
            for xvalues, funcvalues in trainloader:
                if self.train_on_gpu:
                    xvalues, funcvalues = xvalues.cuda(), funcvalues.cuda()

                optimizer.zero_grad()
                #               we sample for every single batch
                #               Setting the batch size here is not ideal
                batch_size = xvalues.shape[0]

                func_idx, contxt_idx = self._get_sample_indexes()
                num_trgt, num_contxt = len(func_idx), len(contxt_idx)

                target_y = funcvalues[:, func_idx, :]
                target_x = xvalues[:, func_idx, :]

                context_y = funcvalues[:, contxt_idx, :]
                context_x = xvalues[:, contxt_idx, :]

                # the encoding is stacked to ensure a one dimensional input
                context_y_stacked = context_y.view(batch_size * num_contxt, -1)
                context_x_stacked = context_x.view(batch_size * num_contxt, -1)

                # running the context values through the encoding
                encoding = encoder(context_x_stacked, context_y_stacked)
                encoding = encoding.view(batch_size, num_contxt, -1)
                # averaging the encoding
                encoding_avg = encoding.mean(1)
                # we need to unsqueeze and repeat the embedding
                # because we need to pair it with every target
                encoding_avg = encoding_avg.unsqueeze(1)
                encoding_exp = encoding_avg.repeat(1, num_trgt, 1)

                encoding_stacked = encoding_exp.view(batch_size * num_trgt, -1)
                target_x_stacked = target_x.view(batch_size * num_trgt, -1)

                decoding = decoder(target_x_stacked, encoding_stacked)
                decoding_rshp = decoding.view(batch_size, num_trgt, -1)

                #
                mu, sigma = decoding_rshp[:, :, 0], decoding_rshp[:, :, 1]

                cov_matrix = Helper.transform_var(sigma)
                distribution = MultivariateNormal(loc=mu, covariance_matrix=cov_matrix)

                loss = distribution.log_prob(target_y.squeeze(-1))
                loss = -torch.mean(loss)
                running_loss += loss

                loss.backward()
                optimizer.step()
            else:
                mean_epoch_loss.append(running_loss / len(trainloader))

                if epoch % self._print_after == 0 and epoch > 0:
                    print(f'Mean loss at epoch {epoch} : {mean_epoch_loss[-1]}')
                    if valiloader:
                        self._validation_run(valiloader, encoder, decoder, epoch)
                        encoder.train(), decoder.train()


        return decoder.state_dict()



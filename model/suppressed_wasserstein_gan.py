"""
Wasserstein GAN class.
"""

import torch
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt

from model.cnn import CNN
from model.transpose_cnn import TransposeCNN
from util.data_utils import tile_images

class SuppressedWassersteinGAN():
    def __init__(self, config):
        # get config args
        self.out_dir = config['output_directory']
        self.batch_size = config['batch_size']
        self.input_dims = config['input_dimensions']
        self.learn_rate = config['learning_rate']
        self.z_dim = config['z_dimension']
        self.name = config['model_name']
        self.verbosity = config['verbosity']

        # initialize logging to create new log file and log any level event
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', \
            filename='{}{}.log'.format(self.out_dir, self.name), filemode='w', \
            level=logging.DEBUG)

        # try to get gpu device, if not just use cpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # initialize critic (CNN) model
        self.critic = CNN(
            in_chan=self.input_dims[-1],
            out_dim=1,
            out_act=torch.nn.LeakyReLU()
        )

        # initialize generator (TransposeCNN) model
        self.generator = TransposeCNN(
            in_dim=self.z_dim,
            out_chan=self.input_dims[-1],
            out_act=torch.nn.Tanh()
        )

        # initialize zdim dimensional normal distribution to sample generator inputs
        self.z_dist = torch.distributions.normal.Normal(
            torch.zeros(self.batch_size, self.z_dim),
            torch.ones(self.batch_size, self.z_dim)
        )

        # initialize bs dimensional uniform distribution to sample eps vals for creating interpolations
        self.eps_dist = torch.distributions.uniform.Uniform(
            torch.zeros(self.batch_size, 1, 1, 1),
            torch.ones(self.batch_size, 1, 1, 1)
        )

        # initialize a random image distriution
        self.img_dist = torch.distributions.Uniform(
            -1.*torch.ones(self.batch_size, 1, 32, 32),
            torch.ones(self.batch_size, 1, 32, 32)
        )

        # sample a batch of z to have constant set of generator inputs as model trains
        self.z_const = self.z_dist.sample()[:64].to(self.device)

        # initialize critic and generator optimizers
        self.crit_opt = torch.optim.Adam(self.critic.parameters(), lr=self.learn_rate)
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.learn_rate)

        # move tensors to device
        self.critic.to(self.device)
        self.generator.to(self.device)

    def print_structure(self):
        print('[INFO] \'{}\' critic structure \n{}'.format(self.name, self.critic))
        print('[INFO] \'{}\' generator structure \n{}'.format(self.name, self.generator))

    def compute_losses(self, real_crit_out, fake_crit_out, crit_grad):
        # compute wasserstein distance estimate
        wass_dist = torch.mean(real_crit_out - fake_crit_out)

        # compute mean of normed critic gradients
        crit_grad_mean_norm = torch.mean(torch.norm(crit_grad, p=2, dim=(2, 3)))

        # lagrangian multiplier for critic gradient penalty (push crit_grad_mean_norm -> 1)
        crit_grad_penalty = (crit_grad_mean_norm - 1.)**2

        # compute generator loss
        gen_loss = wass_dist

        # compute critic loss with lambda=10 weighted critic gradient penalty
        crit_loss = (10.0 * crit_grad_penalty) - wass_dist

        return gen_loss, crit_loss, wass_dist, crit_grad_penalty

    def generate_samples_and_tile(self, z):
        # geneatr a batch of fak images from z input
        fake_img_batch = self.generator(z)

        # detach, move to cpu, and covert images to numpy
        fake_img_batch = fake_img_batch.detach().cpu().numpy()

        # move channel dim to last dim of tensor
        fake_img_batch = np.transpose(fake_img_batch, [0, 2, 3, 1])

        # construct tiled image (squeeze to remove channel dim for grayscale)
        fake_img_tiled = np.squeeze(tile_images(fake_img_batch))

        return fake_img_tiled

    def train(self, dataloader, num_epochs):
        # iterate through epochs
        for e in range(num_epochs):

            # accumulator for wasserstein distance over an epoch
            running_w_dist = 0.0

            # iteate through batches
            for i, batch in enumerate(dataloader):

                # get images from batch
                real_img_batch = batch['image'].to(self.device)

                # get number of samples in batch
                bs = real_img_batch.shape[0]

                # sample from z and eps distribution and clip based on sumber of samples in batch
                z_sample = self.z_dist.sample()[:bs].to(self.device)
                eps_sample = self.eps_dist.sample()[:bs].to(self.device)

                # generate batch of fake images by feeding sampled z through generator
                fake_img_batch = self.generator(z_sample)

                # compute batch of images by interpolating eps_sample amount between real and fake
                # (generated) images
                int_img_batch = (eps_sample * real_img_batch) + \
                                ((1. - eps_sample) * fake_img_batch)

                # compute critic outputs from real, fake, and interpolated image batches
                real_crit_out = self.critic(real_img_batch)
                fake_crit_out = self.critic(fake_img_batch)
                int_crit_out = self.critic(int_img_batch)

                # compute gradient of critic output w.r.t interpolated image inputs
                crit_grad = torch.autograd.grad(
                    outputs=int_crit_out,
                    inputs=int_img_batch,
                    grad_outputs=torch.ones_like(int_crit_out),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                # compute losses
                gen_loss, crit_loss, w_dist, grad_pen = self.compute_losses(
                    real_crit_out,
                    fake_crit_out,
                    crit_grad
                )

                # add regularizer to supress output given random images
                rand_crit_out = self.critic(self.img_dist.sample().to(self.device))
                crit_supp_loss = torch.nn.functional.mse_loss(rand_crit_out, \
                    torch.zeros_like(rand_crit_out))
                crit_loss += 1. * crit_supp_loss

                # NOTE: Currently must update critic and generator separately.
                # If both are updated within the same loop, either updating
                # doesn't happen, or an inplace operator error occurs which
                # prevents gradient computation, depending on the ordering of
                # the zero_grad(), backward(), step() calls. ???

                if i % 10 == 9:
                    # update just the generator (every 10th step)
                    self.gen_opt.zero_grad()
                    gen_loss.backward()
                    self.gen_opt.step()

                    if self.verbosity:
                        # generate const batch of fake samples and tile
                        fake_img_tiled = self.generate_samples_and_tile(self.z_const)

                        # save tiled image
                        plt.imsave('{}{}_gen_step_{}.png'.format(self.out_dir, self.name, \
                            (e*(int(self.conf.ntrain/self.batch_size)+1))+i), fake_img_tiled)
                else:
                    # update just the critic
                    self.crit_opt.zero_grad()
                    crit_loss.backward()
                    self.crit_opt.step()

                # accumulate running wasserstein distance
                running_w_dist += w_dist.item()

            # done with current epoch

            # compute average wasserstein distance over epoch
            epoch_avg_w_dist = running_w_dist / i

            # log epoch stats info
            logging.info('| epoch: {:3} | wasserstein distance: {:6.2f} | gradient penalty: '\
                '{:6.2f} | critic suppression loss: {:6.2f} |'.format(e+1, epoch_avg_w_dist, \
                grad_pen, crit_supp_loss))

            # new sample from z dist
            z_sample = self.z_dist.sample()[:64].to(self.device)

            # generate const batch of fake samples and tile
            fake_img_tiled = self.generate_samples_and_tile(z_sample)

            if self.verbosity:
                # save tiled image
                plt.imsave('{}{}_gen_epoch_{}.png'.format(self.out_dir, self.name, e+1), \
                    fake_img_tiled)

            # save current state of generator and critic
            torch.save(self.generator.state_dict(), '{}{}_generator.pt'.format(self.out_dir, \
                self.name))
            torch.save(self.critic.state_dict(), '{}{}_critic.pt'.format(self.out_dir, \
                self.name))

        # done with all epochs

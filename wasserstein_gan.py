"""
Wasserstein GAN class.
"""

import torch
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

# relative imports
from cnn import CNN
from transpose_cnn import TransposeCNN

from data_utils import tile_images

class WassersteinGAN():
    def __init__(self, config):
        # get args
        self.config = config

        # try to get gpu device, if not just use cpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # initialize critic (CNN) model
        self.critic = CNN(
            in_chan=self.config.nchan,
            out_dim=1,
            out_act=None
        )

        # initialize generator (TransposeCNN) model
        self.generator = TransposeCNN(
            in_dim=self.config.zdim,
            out_chan=self.config.nchan,
            out_act=torch.nn.Tanh()
        )

        # initialize zdim dimensional normal distribution to sample generator inputs
        self.z_dist = torch.distributions.normal.Normal(
            torch.zeros(self.config.bs, self.config.zdim),
            torch.ones(self.config.bs, self.config.zdim)
        )

        # initialize bs dimensional uniform distribution to sample eps vals for creating interpolations
        self.eps_dist = torch.distributions.uniform.Uniform(
            torch.zeros(self.config.bs, 1, 1, 1),
            torch.ones(self.config.bs, 1, 1, 1)
        )

        # sample a batch of z to have constant set of generator inputs as model trains
        self.z_const = self.z_dist.sample().to(self.device)

        # initialize critic and generator optimizers
        self.crit_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr)
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.config.lr)

        # move tensors to device
        self.critic.to(self.device)
        self.generator.to(self.device)

    def print_structure(self):
        print('[INFO] critic structure \n{}'.format(self.critic))
        print('[INFO] generator structure \n{}'.format(self.generator))

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

    def train(self, dataloader):
        print('[INFO] training...')

        # iterate through epochs
        for e in range(self.config.ne):

            # accumulator for wasserstein distance over an epoch
            running_w_dist = 0.0

            # iteate through batches
            for i, batch in tqdm(
                    enumerate(dataloader),
                    desc='[PROGRESS] epoch {}'.format(e+1),
                    total=int(self.config.ntrain/self.config.bs)+1
                ):

                # get images from batch
                real_img_batch = batch[0].to(self.device)

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

                # NOTE: Currently must update critic and generator separately. If both are updated
                # within the same loop, either updatin doesn't happen, or an inplace operator
                # error occurs which prevents gradient computation, depending on the ordering of
                # the zero_grad(), backward(), step() calls. Currently don't know why :(

                if i % 10 == 9:
                    # update just the generator (every 10th step)
                    self.gen_opt.zero_grad()
                    gen_loss.backward()
                    self.gen_opt.step()

                    # generate const batch of fake samples and tile
                    fake_img_tiled = self.generate_samples_and_tile(self.z_const)

                    # save tiled image
                    plt.imsave('/tmp/gen_img_step_{}.png'.format((e*(int(self.config.ntrain/self.config.bs)+1))+i), fake_img_tiled)
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

            # report epoch stats
            print('[INFO] epoch: {}, wasserstein distance: {:.2f}, gradient penalty: {:.2f}'.format(e+1, epoch_avg_w_dist, grad_pen))

            # new sample from z dist
            z_sample = self.z_dist.sample()[:64].to(self.device)

            # generate const batch of fake samples and tile
            fake_img_tiled = self.generate_samples_and_tile(z_sample)

            # save tiled image
            plt.imsave('/tmp/gen_img_epoch_{}.png'.format(e+1), fake_img_tiled)

            # save current state of generator and critic
            torch.save(self.generator.state_dict(), '/tmp/generator.pt')
            torch.save(self.critic.state_dict(), '/tmp/critic.pt')

        # done with all epochs

        print('[INFO] done training')

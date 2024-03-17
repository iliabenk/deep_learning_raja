import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from data_handler import get_inception_score
from itertools import chain
from torchvision import utils

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

class DCGAN_MODEL(object):
    def __init__(self, args):
        print("DCGAN model initalization.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()
        self.D_losses = []
        self.G_losses = []

        self.cuda = False
        self.cuda_index = 0
        # check if cuda is available
        self.check_cuda(args.cuda)

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.number_of_images = 10
        self.number_of_iterations = 0

    # cuda support
    def check_cuda(self, cuda_flag=False):
        print(f'cuda flag is {cuda_flag}')
        if cuda_flag:
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCELoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ")
            print(self.cuda)


    def train(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0
        #self.file = open("inception_score_graph.txt", "w")

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, 100, 1, 1))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)

                if self.cuda:
                    images, z = Variable(images).cuda(self.cuda_index), Variable(z).cuda(self.cuda_index)
                    real_labels, fake_labels = Variable(real_labels).cuda(self.cuda_index), Variable(fake_labels).cuda(self.cuda_index)
                else:
                    images, z = Variable(images), Variable(z)
                    real_labels, fake_labels = Variable(real_labels), Variable(fake_labels)


                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D(images)
                d_loss_real = self.loss(outputs.flatten(), real_labels)
                real_score = outputs

                # Compute BCE Loss using fake images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                # Compute loss with fake images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.loss(outputs.flatten(), real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1

                if generator_iter % 100 == 0:
                    print('Epoch-{}'.format(epoch + 1))
                    self.save_model()

                    if not os.path.exists('dcgan_training_result_images/'):
                        os.makedirs('dcgan_training_result_images/')

                    # Denormalize images and save them in grid 8x8
                    if self.cuda:
                        z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                    else:
                        z = Variable(torch.randn(800, 100, 1, 1))

                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:64]
                    grid = utils.make_grid(samples)
                    utils.save_image(grid, 'dcgan_training_result_images/img_generator_iter_{}.png'.format(str(generator_iter).zfill(3)))

                    time = t.time() - self.t_begin
                    #print("Inception score: {}".format(inception_score))
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(time))

                    # Write to file inception_score, gen_iters, time
                    #output = str(generator_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                    #self.file.write(output)

                self.D_losses.append(d_loss.data)
                self.G_losses.append(g_loss.data)
                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

                    if self.cuda:
                        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                    else:
                        z = Variable(torch.randn(self.batch_size, 100, 1, 1))




        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.number_of_iterations = generator_iter
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        self.G.eval()
        if self.cuda:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        else:
            z = Variable(torch.randn(self.batch_size, 100, 1, 1))

        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dcgan_model_image.png'.")
        utils.save_image(grid, 'dcgan_model_image.png')

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread('dcgan_model_image.png')
        imgplot = plt.imshow(img)
        plt.show()

        labels = random.sample(range(len(test_loader.dataset.classes)), 2)

        dcgan_images = []
        real_images = []

        for batch, (images, targets) in enumerate(test_loader):
            for image, target in zip(images, targets):
                if target.item() in labels:
                    z_dcgan = torch.randn(1, 100, 1, 1)
                    dcgan_image = self.G(z_dcgan).detach().squeeze().numpy()
                    dcgan_images.append(dcgan_image)

                # Select real images
                real_images.append(image.numpy())

                if len(dcgan_images) >= 2:
                    break

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
        ax = axes.ravel()
        ax[0].imshow(dcgan_images[0])
        ax[0].set_title(f'dcgan image 1 of labels {labels}'), ax[0].axis('off')
        ax[1].imshow(dcgan_images[1])
        ax[1].set_title(f'dcgan image 2 of labels {labels}'), ax[1].axis('off')
        ax[2].imshow(real_images[2])
        ax[2].set_title(f'real image 1 of labels {labels}'), ax[2].axis('off')
        ax[3].imshow(real_images[3])
        ax[3].set_title(f'real image 2 of labels {labels}'), ax[3].axis('off')
        plt.tight_layout()
        plt.show()
        return dcgan_images, real_images

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './dcgan_generator.pkl')
        torch.save(self.D.state_dict(), './dcgan_discriminator.pkl')
        print('Models save to ./dcgan_generator.pkl & ./dcgan_discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path, map_location=torch.device('cpu')), strict=False)
        self.G.load_state_dict(torch.load(G_model_path, map_location=torch.device('cpu')), strict=False)
        self.D.eval()
        self.G.eval()
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def generate_latent_walk(self, number):
        if not os.path.exists('dcgan_interpolated_images/'):
            os.makedirs('dcgan_interpolated_images/')

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C, 32, 32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, 'dcgan_interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images to dcgan_interpolated_images/interpolated_{}.".format(str(number).zfill(3)))


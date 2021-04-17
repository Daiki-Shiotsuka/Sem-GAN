
import argparse
import os
import sys
import itertools
import math
import datetime
import time
import cv2
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import datasets

from model import VGGNet, FCNs

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
from PIL import Image

from params import create_parser
from model import CycleGenerator, Discriminator
from lr_helpers import get_lambda_rule
from dataset import CycleGANDataset


def train_loop(opts):

    if opts.image_height == 128:
        res_blocks = 6
    elif opts.image_height >= 256:
        res_blocks = 9

    # GAN networks
    G_AB = CycleGenerator(opts.a_channels, opts.b_channels, res_blocks).to(device)
    G_BA = CycleGenerator(opts.b_channels, opts.a_channels, res_blocks).to(device)
    D_A = Discriminator(opts.a_channels, opts.d_conv_dim).to(device)
    D_B = Discriminator(opts.b_channels, opts.d_conv_dim).to(device)
    # load pretrained segementation networks
    vgg_model_A = VGGNet(requires_grad=True, remove_fc=True).to(device)
    vgg_model_B = VGGNet(requires_grad=True, remove_fc=True).to(device)
    fcn_model_A = FCNs(pretrained_net=vgg_model_A, n_class=20).to(device)
    fcn_model_B = FCNs(pretrained_net=vgg_model_B, n_class=20).to(device)
    fcn_model_A.load_state_dict(torch.load("pretrained/"+opts.dataset_name+"/domainA/latest.pth").module.state_dict())
    fcn_model_B.load_state_dict(torch.load("pretrained/"+opts.dataset_name+"/domainB/latest.pth").module.state_dict())
    fcn_model_A.eval()
    fcn_model_B.eval()


    # Create losses
    criterion_gan = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_semseg = nn.BCEWithLogitsLoss()

    # Create optimizers
    g_optimizer = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                    lr=opts.lr, betas=(opts.beta1, opts.beta2))
    d_a_optimizer = torch.optim.Adam(D_A.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    d_b_optimizer = torch.optim.Adam(D_B.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

    s_a_optimizer = torch.optim.Adam(fcn_model_A.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    s_b_optimizer = torch.optim.Adam(fcn_model_B.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

    # Create learning rate update schedulers
    LambdaLR = get_lambda_rule(opts)
    g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=LambdaLR)
    d_a_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_a_optimizer, lr_lambda=LambdaLR)
    d_b_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_b_optimizer, lr_lambda=LambdaLR)

    s_a_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(s_a_optimizer, lr_lambda=LambdaLR)
    s_b_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(s_b_optimizer, lr_lambda=LambdaLR)

    # Image transformations
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                   ])

    train_dataloader = DataLoader(CycleGANDataset(opts.dataset_name, transform,seg_channels=opts.seg_channels), batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    test_dataloader = DataLoader(CycleGANDataset(opts.dataset_name, transform, mode='val'), batch_size=5, shuffle=True, num_workers=1)



    end_epoch = opts.epochs + opts.start_epoch
    total_batch = len(train_dataloader) * opts.epochs

    for epoch in range(opts.start_epoch, end_epoch):
        for index, batch in enumerate(train_dataloader):
            # Create adversarial target
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            fake_A, fake_B = G_BA(real_B), G_AB(real_A)

            label_A = Variable(batch['lA'].to(device))
            label_B = Variable(batch['lB'].to(device))

            # Train discriminator A
            d_a_optimizer.zero_grad()

            patch_real = D_A(real_A)
            loss_a_real = criterion_gan(patch_real, torch.tensor(1.0).expand_as(patch_real).to(device))
            patch_fake = D_A(fake_A)
            loss_a_fake = criterion_gan(patch_fake, torch.tensor(0.0).expand_as(patch_fake).to(device))
            loss_d_a = loss_a_real + loss_a_fake
            loss_d_a.backward(retain_graph=True)
            d_a_optimizer.step()

            # Train discriminator B
            d_b_optimizer.zero_grad()

            patch_real = D_B(real_B)
            loss_b_real = criterion_gan(patch_real, torch.tensor(1.0).expand_as(patch_real).to(device))
            patch_fake = D_B(fake_B)
            loss_b_fake = criterion_gan(patch_fake, torch.tensor(0.0).expand_as(patch_fake).to(device))
            loss_d_b = loss_b_real + loss_b_fake
            loss_d_b.backward(retain_graph=True)
            d_b_optimizer.step()

            # Train Segmentation
            s_b_optimizer.zero_grad()
            semseg_fake_B = fcn_model_B(fake_B*128)
            loss_semseg_ab = criterion_semseg(semseg_fake_B, label_A)
            #semseg_real_A = fcn_model_A(real_A*128)
            #loss_semseg_ab = criterion_semseg(semseg_fake_B, semseg_real_A)
            loss_semseg_ab.backward(retain_graph=True)
            s_b_optimizer.step()

            s_a_optimizer.zero_grad()
            semseg_fake_A = fcn_model_A(fake_A*128)
            loss_semseg_ba = criterion_semseg(semseg_fake_A, label_B)
            #semseg_real_B = fcn_model_B(real_B*128)
            #loss_semseg_ba = criterion_semseg(semseg_fake_A, semseg_real_B)
            loss_semseg_ba.backward(retain_graph=True)
            s_a_optimizer.step()

            # Train generator

            g_optimizer.zero_grad()
            fake_A, fake_B = G_BA(real_B), G_AB(real_A)
            reconstructed_A, reconstructed_B = G_BA(fake_B), G_AB(fake_A)
            # GAN loss
            patch_a = D_A(fake_A)
            loss_gan_ba = criterion_gan(patch_a, torch.tensor(1.0).expand_as(patch_a).to(device))
            patch_b = D_B(fake_B)
            loss_gan_ab = criterion_gan(patch_b, torch.tensor(1.0).expand_as(patch_b).to(device))
            loss_gan = loss_gan_ab + loss_gan_ba

            # Cycle loss
            loss_cycle_a = criterion_cycle(reconstructed_A, real_A)
            loss_cycle_b = criterion_cycle(reconstructed_B, real_B)
            loss_cycle = (loss_cycle_a + loss_cycle_b) * opts.lambda_cycle # 10

            # Identity loss
            loss_id_a = criterion_identity(G_BA(real_A), real_A)
            loss_id_b = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_a + loss_id_b)* opts.lambda_cycle* 0.5

            # Semantic Segmentaion loss


            semseg_fake_A = fcn_model_A(fake_A*128)
            semseg_fake_B = fcn_model_B(fake_B*128)

            loss_semseg_ab = criterion_semseg(semseg_fake_B, label_A)
            loss_semseg_ba = criterion_semseg(semseg_fake_A, label_B)
            # the rate of considering segmentation loss
            if epoch < 20:
                lambda_semseg = 0.0
            elif epoch < 50:
                lambda_semseg = 0.1
            else:
                lambda_semseg = 0.2

            loss_semseg = (loss_semseg_ab + loss_semseg_ba)*lambda_semseg


            # total loss
            loss_g = loss_gan + loss_cycle + loss_identity + loss_semseg
            loss_g.backward()
            g_optimizer.step()
            if index % 10 == 0:
                print(f"\r[Epoch {epoch+1}/{opts.epochs-opts.start_epoch}] [Index {index}/{len(train_dataloader)}] [D_A loss: {loss_d_a.item():.4f}] [D_B loss: {loss_d_b.item():.4f}] [G loss: adv: {loss_gan.item():.4f}, cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}, semseg: {loss_semseg.item():.4f}]")


        save_sample(G_AB, G_BA, epoch+1, opts, test_dataloader)

        # Update learning reate
        g_lr_scheduler.step()
        d_a_lr_scheduler.step()
        d_b_lr_scheduler.step()
        s_a_lr_scheduler.step()
        s_b_lr_scheduler.step()
        if epoch % opts.checkpoint_every == 0:
            torch.save(G_AB.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_G_AB.pth')
            torch.save(G_BA.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_G_BA.pth')
            torch.save(D_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_D_A.pth')
            torch.save(D_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_D_B_.pth')
            torch.save(G_AB.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_G_AB.pth')
            torch.save(G_BA.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_G_BA.pth')
            torch.save(D_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_D_A.pth')
            torch.save(D_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_D_B.pth')
            torch.save(fcn_model_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_S_A_.pth')
            torch.save(fcn_model_A.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_S_A.pth')
            torch.save(fcn_model_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/{epoch}_S_B_.pth')
            torch.save(fcn_model_B.state_dict(), f'{opts.checkpoint_dir}/{opts.dataset_name}/latest_S_B.pth')

        # save sagmentation samples_cyclegan
        '''
        semseg_real_A = fcn_model_A(real_A*128)
        semseg_real_B = fcn_model_B(real_B*128)
        semseg_fake_B = fcn_model_B(fake_B*128)
        semseg_fake_A = fcn_model_A(fake_A*128)

        semseg_real_A = semseg_real_A.data.cpu().numpy()
        semseg_real_B = semseg_real_B.data.cpu().numpy()
        semseg_fake_B = semseg_fake_B.data.cpu().numpy()
        semseg_fake_A = semseg_fake_A.data.cpu().numpy()

        N, _, h, w = semseg_real_A.shape
        semseg_real_A = semseg_real_A.transpose(0, 2, 3, 1).reshape(-1, 20).argmax(axis=1).reshape(N, h, w)
        semseg_real_B = semseg_real_B.transpose(0, 2, 3, 1).reshape(-1, 20).argmax(axis=1).reshape(N, h, w)
        semseg_fake_B = semseg_fake_B.transpose(0, 2, 3, 1).reshape(-1, 20).argmax(axis=1).reshape(N, h, w)
        semseg_fake_A = semseg_fake_A.transpose(0, 2, 3, 1).reshape(-1, 20).argmax(axis=1).reshape(N, h, w)

        semseg_real_A = semseg_real_A * 10
        semseg_real_B = semseg_real_B * 10
        semseg_fake_B = semseg_fake_B * 10
        semseg_fake_A = semseg_fake_A * 10

        cv2.imwrite(f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_A_real_seg.png", semseg_real_A[0])
        cv2.imwrite(f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_B_real_seg.png", semseg_real_B[0])
        cv2.imwrite(f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_B_fake_seg.png", semseg_fake_B[0])
        cv2.imwrite(f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_A_fake_seg.png", semseg_fake_A[0])

        save_image(real_A, f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_A_real.png", normalize=True)
        save_image(real_B, f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_B_real.png", normalize=True)
        save_image(fake_B, f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_B_fake.png", normalize=True)
        save_image(fake_A, f"{opts.sample_seg_dir}/{opts.dataset_name}/{epoch+1}_A_fake.png", normalize=True)
        '''

def save_sample(G_AB, G_BA, batch, opts, test_dataloader):
    images = next(iter(test_dataloader))
    real_A = Variable(images['A'].to(device))
    real_B = Variable(images['B'].to(device))
    fake_A = G_BA(real_B)
    fake_B = G_AB(real_A)

    #reconstructed_A = G_BA(fake_B)
    #reconstructed_B = G_AB(fake_A)
    image_sample = torch.cat((real_A.data, fake_B.data,
                              real_B.data, fake_A.data
                              #,reconstructed_A.data, reconstructed_B.data
                              ), 0)
    save_image(image_sample, f"{opts.sample_dir}/{opts.dataset_name}/{batch}.png", nrow=5, normalize=True)



def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(f"{opts.sample_dir}/{opts.dataset_name}", exist_ok=True)
    os.makedirs(f"{opts.checkpoint_dir}/{opts.dataset_name}", exist_ok=True)
    os.makedirs(f"{opts.sample_seg_dir}/{opts.dataset_name}", exist_ok=True)

    if opts.load:
        opts.sample_dir = '{}_pretrained'.format(opts.sample_dir)
        opts.sample_every = 20

    print_opts(opts)
    train_loop(opts)

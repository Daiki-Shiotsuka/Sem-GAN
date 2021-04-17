import argparse
import os
import sys
import itertools
import math
import datetime
import time
import cv2

import torchvision.transforms as transforms
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

def test_loop(opts):

    if opts.image_height == 128:
        res_blocks = 6
    elif opts.image_height >= 256:
        res_blocks = 9

    transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                   ])

    # Create networks
    G_AB = CycleGenerator(opts.a_channels, opts.b_channels, res_blocks).to(device)
    G_BA = CycleGenerator(opts.b_channels, opts.a_channels, res_blocks).to(device)
    D_A = Discriminator(opts.a_channels, opts.d_conv_dim).to(device)
    D_B = Discriminator(opts.b_channels, opts.d_conv_dim).to(device)

    G_AB.load_state_dict(torch.load("checkpoints_semgan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_G_AB.pth"))
    G_BA.load_state_dict(torch.load("checkpoints_semgan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_G_BA.pth"))

    vgg_model_A = VGGNet(requires_grad=True, remove_fc=True).to(device)
    vgg_model_B = VGGNet(requires_grad=True, remove_fc=True).to(device)
    fcn_model_A = FCNs(pretrained_net=vgg_model_A, n_class=20).to(device)
    fcn_model_B = FCNs(pretrained_net=vgg_model_B, n_class=20).to(device)
    #fcn_model_A.load_state_dict(torch.load("pretrained/"+opts.dataset_name+"/domainA/latest.pth").module.state_dict())
    #fcn_model_B.load_state_dict(torch.load("pretrained/"+opts.dataset_name+"/domainB/latest.pth").module.state_dict())
    fcn_model_A.load_state_dict(torch.load("checkpoints_semgan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_S_A.pth"))
    fcn_model_B.load_state_dict(torch.load("checkpoints_semgan/"+opts.dataset_name+"/"+str(opts.test_epoch)+"_S_B.pth"))

    fcn_model_A.eval()
    fcn_model_B.eval()

    test_dataloader = DataLoader(CycleGANDataset(opts.dataset_name, transform, mode='test'), batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        if not os.path.exists("result/" + opts.dataset_name):
            os.makedirs("result/" + opts.dataset_name)
        for index, batch in enumerate(test_dataloader):
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            fake_B2A = G_BA(real_B)
            fake_A2B = G_AB(real_A)

            rec_A, rec_B = G_BA(fake_A2B), G_AB(fake_B2A)

            save_image(real_A, f"result/{opts.dataset_name}/{str(index)}_A_real.png",normalize=True)
            save_image(real_B, f"result/{opts.dataset_name}/{str(index)}_B_real.png",normalize=True)
            save_image(fake_A2B, f"result/{opts.dataset_name}/{str(index)}_A2B_fake.png",normalize=True)
            save_image(fake_B2A, f"result/{opts.dataset_name}/{str(index)}_B2A_fake.png",normalize=True)
            save_image(rec_A, f"result/{opts.dataset_name}/{str(index)}_A_rec.png",normalize=True)
            save_image(rec_B, f"result/{opts.dataset_name}/{str(index)}_B_rec.png",normalize=True)

            # save segmentation results
            '''
            semseg_real_A = fcn_model_A(real_A*128)
            semseg_fake_B = fcn_model_B(fake_A2B*128)
            semseg_real_B = fcn_model_B(real_B*128)
            semseg_fake_A = fcn_model_A(fake_B2A*128)

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

            cv2.imwrite(f"result/{opts.dataset_name}/{str(index)}_A_real_seg.png", semseg_real_A[0])
            cv2.imwrite(f"result/{opts.dataset_name}/{str(index)}_B_real_seg.png", semseg_real_B[0])
            cv2.imwrite(f"result/{opts.dataset_name}/{str(index)}_B_fake_seg.png", semseg_fake_B[0])
            cv2.imwrite(f"result/{opts.dataset_name}/{str(index)}_A_fake_seg.png", semseg_fake_A[0])
            '''
            print(str(index))


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')

    test_loop(opts)

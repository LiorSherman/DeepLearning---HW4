import unittest
import os
import sys
import pathlib
import urllib
import shutil
import re
import zipfile

import numpy as np
import torch
import matplotlib.pyplot as plt
from project.ganCollections import *
import cs236781.plot as plot
import cs236781.download
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import IPython.display
import tqdm
from project.gan import save_checkpoint
import math

DATA_DIR = pathlib.Path().absolute().joinpath('project/pytorch-datasets')
DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-bush.zip'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64

Unlimited = False
num_epochs = 99999 if Unlimited is False else math.inf
def loadModels(train=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ganModelNames = ['bgan', 'sngan', 'wgan']
    ganModels = dict()
    if not train:
        for name in ganModelNames:
            modelPath = pathlib.Path().parent.absolute().joinpath(f'project/models/{name}_final')
            ganModels[name] = torch.load(f'{modelPath}.pt', map_location=device)
    else:
        _, dataset_dir = cs236781.download.download_data(out_path=DATA_DIR, url=DATA_URL, extract=True, force=False)

        tf = T.Compose([
            # Resize to constant spatial dimensions
            T.Resize((image_size, image_size)),
            # PIL.Image -> torch.Tensor
            T.ToTensor(),
            # Dynamic range [0,1] -> [-1, 1]
            T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
        ])

        ds_gwb = ImageFolder(os.path.dirname(dataset_dir), tf)

        x0, y0 = ds_gwb[0]
        x0 = x0.unsqueeze(0).to(device)

        batch_size = DLparams['batch_size']
        im_size = ds_gwb[0][0].shape

        dl_train = DataLoader(ds_gwb, batch_size, shuffle=True)

        #constract new models
        ganModels = {'bgan': Gan(im_size, device=device),
                     'sngan': SnGan(im_size, device=device),
                     'wgan': WGan(im_size, device=device)
                    }
        if Unlimited:
            ganModels = {ganName: torch.load(
                    pathlib.Path().parent.absolute().joinpath(f'project/models/GANs/models/{ganName}.pt', map_location=device))
                    for ganName in ganModels}
        #train
        try:
            checkpoint_file = {}
            for ganName, ganModule in ganModels.items():
                checkpoint_file[ganName] = pathlib.Path().parent.absolute().joinpath(
                    f'project/models/{ganName}')  # f'project/checkpoints/{ganName}'

            dsc_avg_losses, gen_avg_losses = {}, {}
            for ganName, ganModule in ganModels.items():
                dsc_avg_losses[ganName] = []
                gen_avg_losses[ganName] = []


            for epoch_idx in range(num_epochs):
                # We'll accumulate batch losses and show an average once per epoch.
                dsc_losses, gen_losses = {}, {}
                for ganName, ganModule in ganModels.items():
                    dsc_losses[ganName] = []
                    gen_losses[ganName] = []
                print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')

                with tqdm.tqdm(total=len(dl_train.batch_sampler), file=sys.stdout) as pbar:
                    for batch_idx, (x_data, _) in enumerate(dl_train):
                        x_data = x_data.to(device)
                        for ganName, ganModule in ganModels.items():
                            dsc_loss, gen_loss = ganModule.trainBatch(x_data)
                            dsc_losses[ganName].append(dsc_loss)
                            gen_losses[ganName].append(gen_loss)
                        pbar.update()
                for ganName, ganModule in ganModels.items():
                    print(f'{ganName}')
                    gen = ganModule.generator
                    dsc_avg_losses[ganName].append(np.mean(dsc_losses[ganName]))
                    gen_avg_losses[ganName].append(np.mean(gen_losses[ganName]))
                    print(f'Discriminator loss: {dsc_avg_losses[ganName][-1]}')
                    print(f'Generator loss:     {gen_avg_losses[ganName][-1]}')
                    if save_checkpoint(gen, dsc_avg_losses[ganName], gen_avg_losses[ganName], checkpoint_file[ganName]):
                       print(f'Saved checkpoint.')

                for ganName, ganModule in ganModels.items():
                    print(f'========{ganName}========')
                    gen = ganModule.generator
                    samples = gen.sample(5, with_grad=False)
                    fig, _ = plot.tensors_as_images(samples.cpu(), figsize=(6, 2))
                    IPython.display.display(fig)
                    plt.close(fig)
                    if epoch_idx%150 == 0 and epoch_idx > 1:
                      IS = inception_score(gen, cuda=True, batch_size=32, resize=True, splits=10)
                      print(f"Inception score for model - is: {IS}")
                    print(f'========================')
            ganModels = {ganName: ganModule.generator for ganName, ganModule in ganModels.items()}
        except KeyboardInterrupt as e:
            print('\n *** Training interrupted by user')

    return ganModels


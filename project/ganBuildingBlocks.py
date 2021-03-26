import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn.utils import spectral_norm
import torch.optim as optim
from project.parameters import *
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import tqdm
import sys

class DiscriminatorX(nn.Module):
    def __init__(self, in_size, spectral_norm_cond=False):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        def spectral_norm_if_true(x, spectral_norm_cond):
            if spectral_norm_cond:
                return spectral_norm(x)
            else:
                return x

        self.encoder = nn.Sequential(
            spectral_norm_if_true(nn.Conv2d(self.in_size[0], 128, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False), spectral_norm_cond),
        )
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.encoder(x).flatten(1)
        # ========================
        return y


class Discriminator(nn.Module):
    def __init__(self, in_size, spectral_norm_cond=False):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        def spectral_norm_if_true(x, spectral_norm_cond):
            if spectral_norm_cond:
                return spectral_norm(x)
            else:
                return x

        self.encoder = nn.Sequential(
            spectral_norm_if_true(nn.Conv2d(self.in_size[0], 128, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False), spectral_norm_cond),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            spectral_norm_if_true(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0, bias=False), spectral_norm_cond),
        )
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.encoder(x).flatten(1)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim=4, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 1024, kernel_size=featuremap_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            z = torch.randn((n, self.z_dim), device=device)
            samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        x = self.decoder(z.view(-1, self.z_dim, 1, 1))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()
    label_noise_delta = label_noise / 2
    y_data_noise = torch.ones(y_data.shape).to(y_data.device)
    y_generated_noise = torch.ones(y_generated.shape).to(y_generated.device)
    y_data_noise.uniform_(data_label - label_noise_delta, data_label + label_noise_delta)
    y_generated_noise.uniform_(1 - data_label - label_noise_delta, 1 - data_label + label_noise_delta)
    loss_data, loss_generated = loss_fn(y_data, y_data_noise), loss_fn(y_generated, y_generated_noise)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()
    generated_data_labels = torch.ones(y_generated.shape).to(y_generated.device) * data_label
    loss = loss_fn(y_generated, generated_data_labels)
    # ========================
    return loss


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    weight = 0.7
    loss_score = lambda loss_a, loss_b: loss_a * weight + loss_b * (1 - weight)
    threshold = loss_score(dsc_losses[-1], gen_losses[-1])
    for dsc_loss, gen_loss in zip(dsc_losses, gen_losses):
        if loss_score(dsc_loss, gen_loss) < threshold:
            torch.save(gen_model, checkpoint_file)
            saved = True
    # ========================

    return saved

def gradient_penalty(data, generated_data, dsc, gamma=10):
    batch_size = data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1)
    epsilon = epsilon.expand_as(data)


    if data.is_cuda:
        epsilon = epsilon.cuda()

    interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
    interpolation = Variable(interpolation, requires_grad=True)

    if data.is_cuda:
        interpolation = interpolation.cuda()

    interpolation_logits = dsc(interpolation)
    grad_outputs = torch.ones(interpolation_logits.size())

    if data.is_cuda:
        grad_outputs = grad_outputs.cuda()

    gradients = torch.autograd.grad(outputs=interpolation_logits,
                              inputs=interpolation,
                              grad_outputs=grad_outputs,
                              create_graph=True,
                              retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    return gamma * ((gradients_norm - 1) ** 2).mean()

def create_optimizer(model_params, opt_params):
    opt_params = opt_params.copy()
    optimizer_type = opt_params['type']
    opt_params.pop('type')
    return optim.__dict__[optimizer_type](model_params, **opt_params)


def inception_score(gen, cuda=True, batch_size=32, resize=False, splits=1, len=50000):
    """ Calculates inception score for a generator. Adapted from https://github.com/sbarratt/inception-score-pytorch
    """
    # N = len(imgs)
    N = len

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Load inception model
    print("Loading inception model")
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=0).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    num_batches = N // batch_size
    print(f"Generating {N} images in {num_batches} batches:")
    with tqdm.tqdm(total=num_batches, file=sys.stdout) as pbar:
        # for i, batch in enumerate(dataloader, 0):
        for i in range(N // batch_size):
            batch = gen.sample(batch_size).cpu()
            batch = batch.type(dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
            pbar.update()
    if N % batch_size != 0: # extra batch
        batch = gen.sample(N % batch_size).cpu()
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[num_batches * batch_size:num_batches * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
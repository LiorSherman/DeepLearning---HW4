from project.ganBuildingBlocks import *


class Gan:
    def __init__(self, in_size,
                 dsc_params=BaselineGanHP['dsc_params'],
                 dsc_optimizer_params=BaselineGanHP['optimizer_params'],
                 gen_params=BaselineGanHP['gen_params'],
                 gen_optimizer_params=BaselineGanHP['optimizer_params'],
                 train_batch_params=BaselineGanHP['train_batch_params'],
                 device='cpu', **kw):
        """
        :param in_size: The size of on input image (without batch dimension).
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        self.discriminator = Discriminator(in_size, dsc_params['spectral_norm_cond']).to(device)
        self.generator = Generator(gen_params['z_dim'], gen_params['featuremap_size'], gen_params['out_channels']).to(device)
        self.dsc_loss_fn = discriminator_loss_fn
        self.gen_loss_fn = generator_loss_fn
        self.dsc_opt = create_optimizer(self.discriminator.parameters(), dsc_optimizer_params)
        self.gen_opt = create_optimizer(self.generator.parameters(), gen_optimizer_params)
        self.with_gradient_penalty = train_batch_params['with_gradient_penalty']

    def discriminator(self):
        return self.discriminator

    def generator(self):
        return self.generator

    def dsc_forward(self, x):
        """
         :param x: Input of shape (N,C,H,W) matching the given in_size.
         :return: Discriminator class score (not probability) of
         shape (N,).
         """
        return self.discriminator(x)

    def gen_forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        return self.generator(z)

    def gen_sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        return self.gen_sample(n, with_grad)

    def disc_loss_fn(self, y_data, y_generated, dscLossParams=BaselineGanHP['dscLossParams']):
        return self.dsc_loss_fn(y_data, y_generated, dscLossParams)

    def gen_loss_fn(self, y_generated, genLossParams=BaselineGanHP['genLossParams']):
        return self.gen_loss_fn(y_generated, genLossParams)

    def train_batch(self, x_data: DataLoader):
        self.dsc_opt.zero_grad()
        generated_data = self.generator.sample(x_data.shape[0])
        real_data_prob = self.discriminator(x_data)
        gen_data_prob = self.discriminator(generated_data)
        grad_penalty = gradient_penalty(x_data, generated_data, self.discriminator) if self.with_gradient_penalty else 0.
        dsc_loss = self.dsc_loss_fn(real_data_prob, gen_data_prob) + grad_penalty
        dsc_loss.backward()
        self.dsc_opt.step()

        self.gen_opt.zero_grad()
        generated_data = self.generator.sample(x_data.shape[0], with_grad=True)
        gen_data_prob = self.discriminator(generated_data)
        gen_loss = self.gen_loss_fn(gen_data_prob)
        gen_loss.backward()
        self.gen_opt.step()

        return dsc_loss.item(), gen_loss.item()


class SnGan(Gan):
    def __init__(self, in_size, device='cpu'):
        super().__init__(in_size, **SNGanHP, device=device)


class WGan(Gan):
    def __init__(self, in_size, device='cpu'):
        super().__init__(in_size, **WGanHP, device=device)
        self.discLossFn = lambda y_data, y_generated: -torch.mean(y_data) + torch.mean(y_generated)
        self.genLossFn = lambda y_generated: -torch.mean(y_generated)

class SnWGan(Gan):
    def __init__(self, in_size, device='cpu'):
        super().__init__(in_size, **SNWGanHP, device=device)
        self.discLossFn = lambda y_data, y_generated: -torch.mean(y_data) + torch.mean(y_generated)
        self.genLossFn = lambda y_generated: -torch.mean(y_generated)
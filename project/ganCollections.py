from project.ganBuildingBlocks import *


class Gan:
    # def __init__(self, in_size, z_dim, featuremap_size=4, out_channels=3, device='cpu', spectral_norm_cond=False, with_gradient_penalty=False, **kw):
    def __init__(self, in_size,
                 dscParams=BaselineGanHP['dscParams'],
                 dscOptimizerParams=BaselineGanHP['optimizerParams'],
                 genParams=BaselineGanHP['genParams'],
                 genOptimizerParams=BaselineGanHP['optimizerParams'],
                 trainBatchParams=BaselineGanHP['trainBatchParams'],
                 device='cpu', **kw):
        """
        :param in_size: The size of on input image (without batch dimension).
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        self.discriminator = Discriminator(in_size, dscParams['spectral_norm_cond']).to(device)
        self.generator = Generator(genParams['z_dim'], genParams['featuremap_size'], genParams['out_channels']).to(device)
        self.discLossFn = discriminator_loss_fn
        self.genLossFn = generator_loss_fn
        self.dscOptimizer = create_optimizer(self.discriminator.parameters(), dscOptimizerParams)
        self.genOptimizer = create_optimizer(self.generator.parameters(), genOptimizerParams)
        self.with_gradient_penalty = trainBatchParams['with_gradient_penalty']

    def discriminator(self):
        return self.discriminator

    def generator(self):
        return self.generator

    def discForward(self, x):
        """
         :param x: Input of shape (N,C,H,W) matching the given in_size.
         :return: Discriminator class score (not probability) of
         shape (N,).
         """
        return self.discriminator(x)

    def genForward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        return self.generator(z)

    def genSample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        return self.genSample(n, with_grad)

    def discLossFn(self, y_data, y_generated, dscLossParams=BaselineGanHP['dscLossParams']):
        return self.discLossFn(y_data, y_generated, dscLossParams)

    def genLossFn(self, y_generated, genLossParams=BaselineGanHP['genLossParams']):
        return self.genLossFn(y_generated, genLossParams)

    def trainBatch(self, x_data: DataLoader):
        self.dscOptimizer.zero_grad()
        generatedData = self.generator.sample(x_data.shape[0])
        realDataProb = self.discriminator(x_data)
        genDataProb = self.discriminator(generatedData)
        gradPenalty = gradient_penalty(x_data, generatedData, self.discriminator) if self.with_gradient_penalty else 0.
        dscLoss = self.discLossFn(realDataProb, genDataProb) + gradPenalty
        dscLoss.backward()
        self.dscOptimizer.step()

        self.genOptimizer.zero_grad()
        generatedData = self.generator.sample(x_data.shape[0], with_grad=True)
        genDataProb = self.discriminator(generatedData)
        genLoss = self.genLossFn(genDataProb)
        genLoss.backward()
        self.genOptimizer.step()

        return dscLoss.item(), genLoss.item()


class SnGan(Gan):
    def __init__(self, in_size, device='cpu'):
        # super().__init__(in_size, z_dim, featuremap_size=featuremap_size, out_channels=out_channels, device=device,
        #                  spectral_norm_cond=True, kw=SNParams)
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
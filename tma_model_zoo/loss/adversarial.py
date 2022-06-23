import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..basics.optimizer import make_optimizer
from ..basics.dynamic_conv import DynamicConv2d


class Adversarial(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.local_discriminator = LocalDiscriminator()

        self.loss_d = 0

        self.optimizer = make_optimizer(args, self.local_discriminator)

    def forward(self, tensors):
        fake, real = tensors['refine'], tensors['gt_depth_hr']

        # Discriminator update
        self.optimizer.zero_grad()

        d_fake = self.local_discriminator(fake.detach())
        d_real = self.local_discriminator(real)
        self.loss_d = 0.5 * ((-(torch.log(d_real + 1e-12) + torch.log(1 - d_fake + 1e-12))).mean())

        self.loss_d.backward()
        self.optimizer.step()

        # Generator update
        d_fake_bp = self.local_discriminator(fake)  # for backpropagation, use fake as it is
        return -torch.log(d_fake_bp + 1e-12).mean()
    
    def calc_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(real_data)

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = self.local_discriminator(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())
        grad_outputs = grad_outputs.to(real_data)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=grad_outputs, create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.local_discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = functional.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = functional.binary_cross_entropy_with_logits(fake, label_fake)
        return bce_real + bce_fake


class LocalDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()

        self.conv1 = DynamicConv2d(1, ndf, stride=2, act=nn.LeakyReLU(inplace=True))
        self.conv2 = DynamicConv2d(ndf * 1, ndf * 2, stride=2, act=nn.LeakyReLU(inplace=True))
        self.conv3 = DynamicConv2d(ndf * 2, ndf * 4, stride=2, act=nn.LeakyReLU(inplace=True))
        self.conv4 = DynamicConv2d(ndf * 4, ndf * 8, stride=1, act=nn.LeakyReLU(inplace=True))
        self.conv5 = DynamicConv2d(ndf * 8, 1, stride=1, act=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return torch.sigmoid(self.conv5(x))

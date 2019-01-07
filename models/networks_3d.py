import math
from torch import nn
import torch
from .basics import get_norm_layer, init_net


def define_G_3D(nz=200, res=128, model='G0', ngf=64, norm='batch3d',
                init_type='xavier', init_param=0.02, gpu_ids=[]):
    if model == 'G0':
        netG = _netG0(bias=False, res=res, nz=nz, ngf=ngf, norm=norm)
    else:
        raise NotImplementedError('3D G [%s] is not implemented' % model)
    return init_net(netG, init_type, init_param, gpu_ids)


def define_D_3D(res=128, model='D0', ndf=64, norm='none',
                init_type='xavier', init_param=0.02, gpu_ids=[]):
    if model == 'D0':
        netD = _netD0(bias=False, res=res, nc=1, ndf=ndf, norm=norm)
    else:
        raise NotImplementedError('3D D [%s] is not implemented' % model)
    return init_net(netD, init_type, init_param, gpu_ids)


def deconvBlock(input_nc, output_nc, bias, norm_layer=None, nl='relu'):
    layers = [nn.ConvTranspose3d(input_nc, output_nc, 4, 2, 1, bias=bias)]

    if norm_layer is not None:
        layers += [norm_layer(output_nc)]
    if nl == 'relu':
        layers += [nn.ReLU(True)]
    elif nl == 'lrelu':
        layers += [nn.LeakyReLU(0.2, inplace=True)]
    else:
        raise NotImplementedError('NL layer {} is not implemented' % nl)
    return nn.Sequential(*layers)


# the last layer of a generator
def toRGB(input_nc, output_nc, bias, zero_mean=False, sig=True):
    layers = [nn.ConvTranspose3d(input_nc, output_nc, 4, 2, 1, bias=bias)]
    if sig:
        layers += [nn.Sigmoid()]
    return nn.Sequential(*layers)


class _netG0(nn.Module):
    def __init__(self, bias, res, nz=200, ngf=64, max_nf=8, nc=1, norm='batch'):
        super(_netG0, self).__init__()
        norm_layer = get_norm_layer(layer_type=norm)
        self.res = res
        self.block_0 = nn.Sequential(*[nn.ConvTranspose3d(nz, ngf * max_nf, 4, 1, 0, bias=bias), norm_layer(ngf * 8), nn.ReLU(True)])
        self.n_blocks = 1
        input_dim = ngf * max_nf
        n_layers = int(math.log(res, 2)) - 3
        for n in range(n_layers):
            input_nc = int(max(ngf, input_dim))
            output_nc = int(max(ngf, input_dim // 2))
            setattr(self, 'block_{:d}'.format(self.n_blocks), deconvBlock(input_nc, output_nc, bias, norm_layer=norm_layer, nl='relu'))
            input_dim /= 2
            self.n_blocks += 1

        setattr(self, 'toRGB_{:d}'.format(res), toRGB(output_nc, nc, bias, sig=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, return_stat=False):
        output = input
        for n in range(self.n_blocks):
            block = getattr(self, 'block_{:d}'.format(n))
            output = block(output)
        toRGB = getattr(self, 'toRGB_{:d}'.format(self.res))
        output = toRGB(output)
        output = output / 2  # HACK
        if return_stat:
            stat = [output.max().item(), output.min().item(), output.std().item(), output.mean().item()]
            return self.sigmoid(output), stat
        else:
            return self.sigmoid(output)


def convBlock(input_nc, output_nc, bias, norm_layer=None):
    layers = [nn.Conv3d(input_nc, output_nc, 4, 2, 1, bias=bias)]
    if norm_layer is not None:
        layers += [norm_layer(output_nc)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


class _netD0(nn.Module):
    def __init__(self, bias=False, res=128, final_res=128, nc=1, ndf=64, max_nf=8, norm='none'):
        super(_netD0, self).__init__()
        self.res = res
        self.n_blocks = 0
        norm_layer = get_norm_layer(layer_type=norm)
        n_layers = int(math.log(res, 2)) - 3
        n_final_layers = int(math.log(final_res, 2)) - 3
        self.offset = n_final_layers - n_layers
        setattr(self, 'fromRGB_{:d}'.format(res), fromRGB(1, ndf * min(2 ** max(0, self.offset - 1), max_nf), bias))
        for n in range(n_final_layers - n_layers, n_final_layers):
            input_nc = ndf * min(2 ** max(0, n - 1), max_nf)
            output_nc = ndf * min(2 ** n, max_nf)
            block_name = 'block_{}'.format(n)
            setattr(self, block_name, convBlock(input_nc, output_nc, bias, norm_layer))
            self.n_blocks += 1
        block_name = 'block_{:d}'.format(n_final_layers)
        setattr(self, block_name, nn.Conv3d(ndf * max_nf, 1, 4, 1, 0, bias=bias))
        self.n_blocks += 1

    def forward(self, input):
        fromRGB = getattr(self, 'fromRGB_{:d}'.format(self.res))
        output = fromRGB(input)
        for n in range(self.n_blocks):
            block = getattr(self, 'block_{:d}'.format(n + self.offset))
            output = block(output)
        return output.view(-1, 1).squeeze(1)


# the first layer of a discriminator
def fromRGB(input_nc, output_nc, bias):
    layers = []
    layers += [nn.Conv3d(input_nc, output_nc, 4, 2, 1, bias=bias)]
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return nn.Sequential(*layers)


def _calc_grad_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

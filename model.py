import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

import functools
from torch.autograd import Variable


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


def activation(input):
    return F.relu(input)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 n_class,
                 padding=1, post=True, resize=True,
                 normalize=True, self_attention=False):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                         padding=padding, bias=False)
        init_conv(conv)
        self.conv = spectral_norm(conv)

        self.resize = resize
        self.post = post
        if self.post:
            self.bn = nn.BatchNorm2d(out_channel, affine=False)

        self.embed = nn.Embedding(n_class, out_channel * 2)
        self.embed.weight.data[:, :out_channel] = 1
        self.embed.weight.data[:, out_channel:] = 0

        self.attention = self_attention
        if self_attention:
            self.query = nn.Conv1d(out_channel, out_channel // 8, 1)
            self.key = nn.Conv1d(out_channel, out_channel // 8, 1)
            self.value = nn.Conv1d(out_channel, out_channel, 1)
            self.gamma = nn.Parameter(torch.tensor(0.0))

            init_conv(self.query)
            init_conv(self.key)
            init_conv(self.value)

    def forward(self, input, class_id=None):
        out = input
        if self.resize:
            out = self.upsample(input)
        out = self.conv(out)
        if self.post:
            out = self.bn(out)
            embed = self.embed(class_id)
            gamma, beta = embed.chunk(2, 1)
            #print(out.shape, gamma.shape, beta.shape)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            out = gamma * out + beta
            out = activation(out)

        if self.attention:
            shape = out.shape
            flatten = out.view(shape[0], shape[1], -1)
            query = self.query(flatten).permute(0, 2, 1)
            key = self.key(flatten)
            value = self.value(flatten)
            #print(key.shape, value.shape)
            query_key = torch.bmm(query, key)
            attn = F.softmax(query_key, 1)
            attn = torch.bmm(value, attn)
            attn = attn.view(*shape)
            #print(out.shape, attn.shape)
            out = self.gamma * attn + out

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=1, bn=False,
                 self_attention=False):
        super().__init__()

        conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                         stride, padding, bias=False)
        init_conv(conv)
        self.conv = spectral_norm(conv)
        self.use_bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(out_channel, affine=True)

        self.attention = self_attention
        if self_attention:
            query = nn.Conv1d(out_channel, out_channel // 8, 1)
            key = nn.Conv1d(out_channel, out_channel // 8, 1)
            value = nn.Conv1d(out_channel, out_channel, 1)
            self.gamma = nn.Parameter(torch.tensor(0.0))

            init_conv(query)
            init_conv(key)
            init_conv(value)

            self.query = spectral_norm(query)
            self.key = spectral_norm(key)
            self.value = spectral_norm(value)

    def forward(self, input):
        out = self.conv(input)
        if self.use_bn:
            out = self.bn(out)
        out = F.leaky_relu(out, negative_slope=0.2)

        if self.attention:
            shape = out.shape
            flatten = out.view(shape[0], shape[1], -1)
            query = self.query(flatten).permute(0, 2, 1)
            key = self.key(flatten)
            value = self.value(flatten)
            query_key = torch.bmm(query, key)
            attn = F.softmax(query_key, 1)
            attn = torch.bmm(value, attn)
            attn = attn.view(*shape)
            out = self.gamma * attn + out

        return out


class Generator(nn.Module):
    def __init__(self, code_dim=100, n_class=10):
        super().__init__()

        self.lin_code = nn.Linear(code_dim, 4 * 4 * 512)
        self.conv1 = UpsampleConvBlock(512, 512, [3, 3], n_class)
        self.conv2 = UpsampleConvBlock(512, 256, [3, 3], n_class)
        self.conv3 = UpsampleConvBlock(256, 256, [3, 3], n_class,
                                       self_attention=True)
        self.conv4 = UpsampleConvBlock(256, 128, [3, 3], n_class)
        self.conv5 = UpsampleConvBlock(128, 64, [3, 3], n_class)
        self.conv6 = UpsampleConvBlock(64, 3, [3, 3], 1,
                                       resize=False, post=False)
        init_linear(self.lin_code)

    def forward(self, input, class_id):
        out = self.lin_code(input)
        out = activation(out)
        out = out.view(-1, 512, 4, 4)
        out = self.conv1(out, class_id)
        out = self.conv2(out, class_id)
        out = self.conv3(out, class_id)
        out = self.conv4(out, class_id)
        out = self.conv5(out, class_id)
        out = self.conv6(out)

        return F.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()

        self.conv = nn.Sequential(ConvBlock(3, 64, [3, 3], 2),
                                  ConvBlock(64, 128, [3, 3], 2),
                                  ConvBlock(128, 256, [3, 3], 2),
                                  ConvBlock(256, 256, [3, 3], 2),
                                  ConvBlock(256, 512, [3, 3], 2),
                                  ConvBlock(512, 512, [3, 3],
                                            self_attention=True))
        linear = nn.Linear(512, 1)
        init_linear(linear)
        self.linear = spectral_norm(linear)

        embed = nn.Embedding(n_class, 512)
        embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(embed)

    def forward(self, input, class_id):
        out = self.conv(input)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.sum(2)
        out_linear = self.linear(out).squeeze()
        embed = self.embed(class_id)
        prod = (out * embed).sum(1)

        return out_linear + prod

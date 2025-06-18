import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
import numpy as np


class RefineGenerator(nn.Module):
    def __init__(self, img_dim, con_dim, cnum):
        super(RefineGenerator, self).__init__()
        input_dim = img_dim + con_dim
        self.conv1 = gen_GatedConv(input_dim, cnum * 1, 5, 1, padding=2, rate=1, norm='in', activation='elu')
        self.conv2 = gen_GatedConv(cnum * 1, cnum * 2, 3, 2, padding=1, rate=1, norm='in', activation='elu')
        self.conv3 = gen_GatedConv(cnum * 2, cnum * 4, 3, 2, padding=1, rate=1, norm='in', activation='elu')
        self.conv4 = gen_GatedConv(cnum * 4, cnum * 8, 3, 2, padding=1, rate=1, norm='in', activation='elu')

        self.conv51_atrous1 = gen_GatedConv(cnum * 8, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv61_atrous1 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv61_atrous2 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv71_atrous1 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv71_atrous2 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv71_atrous4 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv81_atrous1 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1,  norm='in', activation='elu')
        self.conv81_atrous2 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv81_atrous4 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv81_atrous8 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=8, rate=8,norm='in', activation='elu')
        self.conv91 = gen_GatedConv(cnum * 8, cnum * 8, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')

        self.conv52_atrous1 = gen_GatedConv(cnum * 8, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv62_atrous1 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv62_atrous2 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv72_atrous1 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1,  norm='in', activation='elu')
        self.conv72_atrous2 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2,  norm='in', activation='elu')
        self.conv72_atrous4 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv82_atrous1 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv82_atrous2 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv82_atrous4 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv82_atrous8 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=8, rate=8, norm='in', activation='elu')
        self.conv92 = gen_GatedConv(cnum * 8, cnum * 8, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')

        self.deconv1 = gen_conv(cnum * 8 * 2, cnum * 8, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv1_conv1 = gen_conv(cnum * 8, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv1_conv2 = gen_conv(cnum * 8 * 2, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv1_gated = gen_GatedConv(cnum * 8 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2 = gen_conv(cnum * 4 * 2, cnum * 4, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv2_conv1 = gen_conv(cnum * 4, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2_conv2 = gen_conv(cnum * 4 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2_gated = gen_GatedConv(cnum * 4 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3 = gen_conv(cnum * 2 * 2, cnum * 2, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv3_conv1 = gen_conv(cnum * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3_conv2 = gen_conv(cnum * 2 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3_gated = gen_GatedConv(cnum * 2 * 2, cnum * 1, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.conv6 = gen_GatedConv(cnum * 1 * 2, cnum // 2, 3, 1, padding=1, rate=1, norm='none', activation='elu')
        self.conv7 = gen_GatedConv(cnum // 2, img_dim, 3, 1, padding=1, rate=1, norm='none', activation='none')

        self.conv8 = gen_GatedConv(img_dim, img_dim, 3, 1, padding=1, rate=1, norm='none', activation='elu')
        self.conv9 = gen_GatedConv(img_dim, img_dim, 3, 1, padding=1, rate=1, norm='none', activation='none')

    def forward(self, x):

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_conv51_dilated1 = self.conv51_atrous1(x_conv4)
        x_conv61_dilated1 = self.conv61_atrous1(x_conv51_dilated1)
        x_conv61_dilated2 = self.conv61_atrous2(x_conv51_dilated1)
        x_conv71_dilated1 = self.conv71_atrous1(torch.cat([x_conv61_dilated1, x_conv61_dilated2], dim=1))
        x_conv71_dilated2 = self.conv71_atrous2(torch.cat([x_conv61_dilated1, x_conv61_dilated2], dim=1))
        x_conv71_dilated4 = self.conv71_atrous4(torch.cat([x_conv61_dilated1, x_conv61_dilated2], dim=1))
        x_conv81_dilated1 = self.conv81_atrous1(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv81_dilated2 = self.conv81_atrous2(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv81_dilated4 = self.conv81_atrous4(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv81_dilated8 = self.conv81_atrous8(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv91 = self.conv91(torch.cat([x_conv81_dilated1, x_conv81_dilated2, x_conv81_dilated4, x_conv81_dilated8], dim=1))

        x_conv52_dilated1 = self.conv52_atrous1(x_conv91)
        x_conv62_dilated1 = self.conv62_atrous1(x_conv52_dilated1)
        x_conv62_dilated2 = self.conv62_atrous2(x_conv52_dilated1)
        x_conv72_dilated1 = self.conv72_atrous1(torch.cat([x_conv62_dilated1, x_conv62_dilated2], dim=1))
        x_conv72_dilated2 = self.conv72_atrous2(torch.cat([x_conv62_dilated1, x_conv62_dilated2], dim=1))
        x_conv72_dilated4 = self.conv72_atrous4(torch.cat([x_conv62_dilated1, x_conv62_dilated2], dim=1))
        x_conv82_dilated1 = self.conv82_atrous1(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv82_dilated2 = self.conv82_atrous2(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv82_dilated4 = self.conv82_atrous4(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv82_dilated8 = self.conv82_atrous8(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv92 = self.conv92(torch.cat([x_conv82_dilated1, x_conv82_dilated2, x_conv82_dilated4, x_conv82_dilated8], dim=1))

        tmp = self.deconv1(torch.cat([x_conv92, x_conv4], dim=1))
        x_deconv1_1 = self.deconv1_conv1(tmp)
        tmp1 = F.interpolate(x_conv92, scale_factor=2, mode='bilinear', align_corners=True)
        tmp2 = F.interpolate(x_conv4, scale_factor=2, mode='bilinear', align_corners=True)
        x_deconv1_2 = self.deconv1_conv2(torch.cat([tmp1, tmp2], dim=1))
        x_deconv1 = self.deconv1_gated(torch.cat([x_deconv1_1, x_deconv1_2], dim=1))

        tmp = self.deconv2(torch.cat([x_deconv1, x_conv3], dim=1))
        x_deconv2_1 = self.deconv2_conv1(tmp)
        tmp1 = F.interpolate(x_deconv1, scale_factor=2, mode='bilinear', align_corners=True)
        tmp2 = F.interpolate(x_conv3, scale_factor=2, mode='bilinear', align_corners=True)
        x_deconv2_2 = self.deconv2_conv2(torch.cat([tmp1, tmp2], dim=1))
        x_deconv2 = self.deconv2_gated(torch.cat([x_deconv2_1, x_deconv2_2], dim=1))

        tmp = self.deconv3(torch.cat([x_deconv2, x_conv2], dim=1))
        x_deconv3_1 = self.deconv3_conv1(tmp)
        tmp1 = F.interpolate(x_deconv2, scale_factor=2, mode='bilinear', align_corners=True)
        tmp2 = F.interpolate(x_conv2, scale_factor=2, mode='bilinear', align_corners=True)
        x_deconv3_2 = self.deconv3_conv2(torch.cat([tmp1, tmp2], dim=1))
        x_deconv3 = self.deconv3_gated(torch.cat([x_deconv3_1, x_deconv3_2], dim=1))

        x_conv6 = self.conv6(torch.cat([x_deconv3, x_conv1], dim=1))
        x_conv7 = self.conv7(x_conv6)  # torch.Size([16, 3, 256, 256])
        x_conv8 = self.conv8(x_conv7)
        x_stage1 = self.conv9(x_conv8)  # torch.Size([16, 3, 256, 256])
        x_stage1 = nn.Tanh()(x_stage1)

        return x_stage1


class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.dis_conv_module(x)
        x_final = self.linear(x4.view(x4.size()[0], -1))

        return x1, x2, x3, x4, x_final


class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.dis_conv_module(x)
        x_final = self.linear(x4.view(x4.size()[0], -1))

        return x1, x2, x3, x4, x_final


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        return x1, x2, x3, x4


def gen_linear(input_dim, output_dim, activation='lrelu'):
    return LinearBlock(input_dim=input_dim, output_dim=output_dim, activation=activation)

def gen_sty_conv(resolution, input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, norm='none',
                 activation='elu', transpose=False, use_noise=True):
    return StyleConv2dBlock(resolution, input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate, weight_norm='wn', norm=norm,
                       activation=activation, transpose=transpose, use_noise=use_noise)

def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, norm='none',
             activation='elu', transpose=False):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate, norm=norm,
                       activation=activation, transpose=transpose)

def gen_GatedConv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, activation='elu', norm='none'):
    return GatedConv2d(input_dim, output_dim,
                 kernel_size, stride=stride,
                 conv_padding=padding, dilation=rate,
                 pad_type='zero',
                 activation=activation, norm=norm, sn=False)

def gen_ResConv(in_channels, out_channels, hidden_channels=None, norm='none', activation='elu', sample_type='none'):
    return ResBlock(in_channels, out_channels, hidden_channels=hidden_channels, norm=norm, activation=activation,
                    sample_type=sample_type)

def gen_MoResConv(in_channels, out_channels, hidden_channels=None, norm='none', activation='elu', sample_type='none'):
    return MoResBlock(in_channels, out_channels, hidden_channels=hidden_channels, norm=norm, activation=activation,
                      sample_type=sample_type)

def gen_AttResConv(in_channels, out_channels, hidden_channels=None, norm='none', activation='elu', sample_type='none'):
    return AttResBlock(in_channels, out_channels, hidden_channels=hidden_channels, norm=norm, activation=activation,
                       sample_type=sample_type)

def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class TimeStepEncoding(nn.Module):
    """位置编码"""
    #num+hiddens:向量长度  max_len:序列最大长度
    def __init__(self, num_hiddens, max_len=50):
        super().__init__()
        # self.dropout = nn.Dropout(drop_rate)
        # 创建一个足够长的P : (1, 1000, 32)
        self.P = torch.zeros((1, max_len, num_hiddens))
        #本例中X的维度为(1000, 16)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)   #::2意为指定步长为2 为[start_index : end_index : step]省略end_index的写法
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, timestep, device, frozen=False):
        if frozen:
            embedding = self.P[:, 0, :].to(device)
        else:
            embedding = self.P[:, timestep-1, :].to(device)
        # embedding = self.dropout(embedding)
        return embedding


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation='lrelu'):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)
        # initialize activation
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation.lower() == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'prelu':
            self.activation = nn.PReLU()
        elif activation.lower() == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


class StyleConv2dBlock(nn.Module):
    def __init__(self, resolution, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False, use_noise=True, modal='text'):
        super(StyleConv2dBlock, self).__init__()
        self.use_bias = True
        self.use_noise = use_noise
        self.resolution = resolution
        self.modal = modal
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)  # inplace=True
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)  # inplace=True
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)  # inplace=True
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)  # inplace=True
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)
        
        if use_noise:
            self.register_buffer('noise_const', torch.randn([1, 1, resolution, resolution]))
            self.noise_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x, miu, sigma, noise_mode='random'):
        batch_size = x.size(0)
        miu = miu.to(self.conv.weight.device)
        sigma = sigma.to(self.conv.weight.device)

        weight = self.conv.weight
        out_channels, in_channels, kh, kw = weight.size()
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        miu = miu / miu.norm(float('inf'), dim=[1,2,3], keepdim=True) # max_Ihw
        sigma = sigma / sigma.norm(float('inf'), dim=[1,2,3], keepdim=True) # max_Ihw

        miu0 = miu.clone().to(x.device)
        sigma0 = sigma.clone().to(x.device)

        miu = miu.mean(dim=[2,3])   # max_I
        sigma = sigma.mean(dim=[2,3])   # max_I

        miu = miu.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
        sigma = sigma.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]

        w = weight.unsqueeze(0) # [NOIkk]
        w = sigma * w + miu # [NOIkk]
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

        self.conv.weight = w.mean(dim=[0])  # [OIkk]

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([batch_size, 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        if self.pad:
            x = self.pad(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
        if noise is not None:
            x = x + noise
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x, miu0, sigma0


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)  # inplace=True
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)  # inplace=True
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)  # inplace=True
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)  # inplace=True
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.pad(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# -----------------------------------------------
#                Res ConvBlock
# -----------------------------------------------
class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None, norm='none', activation='elu',
                 sample_type='none'):
        super(ResBlock, self).__init__()

        hidden_channels = out_channels // 2 if hidden_channels is None else hidden_channels
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        self.conv1 = gen_conv(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                              activation=activation, transpose=False)
        self.conv2 = gen_conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                              activation=activation, transpose=False)
        self.bypass = gen_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                               activation=activation, transpose=False)

        self.model = nn.Sequential(self.conv1, self.conv2, )
        self.shortcut = nn.Sequential(self.bypass, )

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out


# -----------------------------------------------
#                Modulation Res ConvBlock
# -----------------------------------------------
class MoResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None, norm='none', activation='elu',
                 sample_type='none'):
        super(MoResBlock, self).__init__()

        hidden_channels = out_channels // 2 if hidden_channels is None else hidden_channels
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        self.conv_y = gen_conv(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                               activation=activation, transpose=False)
        self.conv_beta = gen_conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm='none',
                                  activation='none', transpose=False)
        self.conv_gamma = gen_conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm='none',
                                   activation='none', transpose=False)
        self.mainpath = gen_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                                 activation=activation, transpose=False)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):  # in this project, x is feature map of the Im, y is the example feature map.
        f_y = self.conv_y(y)
        beta = self.conv_beta(f_y)
        gamma = self.conv_gamma(f_y)
        f_x = self.mainpath(x)
        out = f_x + (gamma * f_x + beta) * self.alpha
        if self.sample:
            out = self.pool(out)

        return out


# -----------------------------------------------
#                Attention Res ConvBlock
# -----------------------------------------------
class AttResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None, norm='none', activation='elu',
                 sample_type='none'):
        super(AttResBlock, self).__init__()

        hidden_channels = out_channels // 2 if hidden_channels is None else hidden_channels
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        self.conv_beta = gen_conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm='none',
                                  activation='none', transpose=False)
        self.conv_y = gen_conv(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                               activation=activation, transpose=False)
        self.query_y = gen_conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                                activation=activation, transpose=False)
        self.query_x = gen_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                                activation=activation, transpose=False)
        self.mainpath = gen_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, rate=1, norm=norm,
                                 activation=activation, transpose=False)
        self.softmax = nn.Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.zeros(1))


    def forward(self, x, y):  # in this project, x is feature map of the Im, y is the example feature map.
        conv_y = self.conv_y(y)
        beta = self.conv_beta(conv_y)
        query_y = self.query_y(conv_y)
        query_x = self.query_x(x)
        B, C, H, W = query_x.size()  # B, out_channels, H, W
        proj_query_x = query_x.view(B, C, -1)  # B, out_channels, HW
        proj_query_y = query_y.view(B, C, -1).permute(0, 2, 1)  # B, HW, out_channels
        energy = torch.bmm(proj_query_y, proj_query_x)  # B, HW, HW
        attention = self.softmax(energy)
        corr_fx = torch.bmm(proj_query_x, attention).view(B, C, H, W)  # B, out_channels, H, W
        f_x = self.mainpath(x)
        out = f_x + (corr_fx + beta) * self.alpha
        if self.sample:
            out = self.pool(out)

        return out


# -----------------------------------------------
#                Gated ConvBlock
# -----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1,
                 padding=0, conv_padding=0,
                 dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(GatedConv2d, self).__init__()
        self.use_bias = True
        # Initialize the padding scheme
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)  # inplace=True
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)  # inplace=True
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)  # inplace=True
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)  # inplace=True
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation, bias=self.use_bias)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation, bias=self.use_bias)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)

        if self.activation:
            x = self.activation(conv) * gated_mask
        else:
            x = conv * gated_mask

        if self.norm:
            x = self.norm(x)

        return x



# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)

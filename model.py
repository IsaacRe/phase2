import torch.nn as nn
from layers import *
from resnet import *


def _block_size(in_dim, out_dim, n_blocks):
    """Returns block_size setting that ensures same memory cost as
        vanilla layer with corresponding in_dim and out_dim"""
    return int(in_dim * out_dim / (in_dim + out_dim) / n_blocks)


def _block_size_conv(in_ch, out_ch, filter_h, filter_w, n_blocks):
    return _block_size(in_ch * filter_h * filter_w, out_ch, n_blocks)


def _block_size_conv3x3(in_ch, out_ch, n_blocks):
    return _block_size_conv(in_ch, out_ch, 3, 3, n_blocks)


def lrm3x3(in_planes, out_planes, n_blocks, block_size, stride=1, cache_attn=False):
    """3x3 low-rank mixture convolution with padding"""
    return LRMConvV1(in_planes, out_planes, 3, n_blocks, block_size,
                     stride=stride, padding=1, cache_attn=cache_attn)


def lrm1x1(in_planes, out_planes, n_blocks, block_size, stride=1, cache_attn=False):
    """1x1 low-rank mixture convolution with padding"""
    return LRMConvV1(in_planes, out_planes, 1, n_blocks, block_size,
                     stride=stride, padding=0, cache_attn=cache_attn)


class LRMBlockV1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 n_blocks=1, cache_attn=False):
        super(LRMBlockV1, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = lrm3x3(inplanes, planes, n_blocks, _block_size_conv3x3(inplanes, planes, n_blocks),
                            stride=stride, cache_attn=cache_attn)
        self.bn1 = norm_layer(planes)
        self.conv2 = lrm3x3(planes, planes, n_blocks, _block_size_conv3x3(planes, planes, n_blocks),
                            cache_attn=cache_attn)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        if downsample is not None:
            downsample = nn.Sequential(
                lrm1x1(downsample[0].in_channels, downsample[0].out_channels, n_blocks,
                       _block_size_conv3x3(downsample[0].in_channels, downsample[0].out_channels, n_blocks),
                       stride=downsample[0].stride, cache_attn=cache_attn),
                norm_layer(downsample[0].out_channels)
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HashBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 detach=False, n_keys=256, cache_attn=False, temperature=1.0,
                 cut_residual=False):
        super(HashBlock, self).__init__()

        ###################################
        self.detach = detach
        self.cut_residual = cut_residual
        ###################################

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        ###################################
        self.hashconv = HashConv(planes, planes, (3, 3), n_keys,
                                 detach=detach, cache_attn=cache_attn, t=temperature)
        ###################################

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        ###################################
        # cut gradient to residual if only training downstream layers
        if self.detach:
            identity = identity.detach()
        ###################################

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hashconv(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        ###################################
        # cut the residual connection if specified
        if not self.cut_residual:
            out += identity
        ###################################

        out = self.relu(out)

        return out


def lrm_resnet18(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return build_resnet('resnet18', LRMBlockV1, [2, 2, 2, 2], False, True,
                        **kwargs)


# test models
if __name__ == '__main__':
    lrm_resnet18()
    resnet18()

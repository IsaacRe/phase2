import torch.nn as nn
from layers import *
from resnet import *


def _block_size(in_dim, out_dim, n_blocks, alpha=1.0):
    """Returns block_size setting that ensures same memory cost as
        vanilla layer with corresponding in_dim and out_dim"""
    return int(in_dim * out_dim / (in_dim + out_dim) / n_blocks * alpha)


def _block_size_conv(in_ch, out_ch, filter_h, filter_w, n_blocks, alpha=1.0):
    return _block_size(in_ch * filter_h * filter_w, out_ch, n_blocks, alpha=alpha)


def _block_size_conv3x3(in_ch, out_ch, n_blocks, alpha=1.0):
    return _block_size_conv(in_ch, out_ch, 3, 3, n_blocks, alpha=alpha)


def lrm3x3(in_planes, out_planes, n_blocks, block_size, stride=1, cache_attn=False):
    """3x3 low-rank mixture convolution with padding"""
    return LRMConvV1(in_planes, out_planes, 3, n_blocks, block_size,
                     stride=stride, padding=1, cache_attn=cache_attn)


def lrm1x1(in_planes, out_planes, n_blocks, block_size, stride=1, cache_attn=False):
    """1x1 low-rank mixture convolution with padding"""
    return LRMConvV1(in_planes, out_planes, 1, n_blocks, block_size,
                     stride=stride, padding=0, cache_attn=cache_attn)


# Modified from BasicBlock - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class LRMBlockV1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 n_blocks=1, block_size_alpha=1.0, cache_attn=False):
        super(LRMBlockV1, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        block_size_conv1 = _block_size_conv3x3(inplanes, planes, n_blocks, alpha=block_size_alpha)
        block_size_conv = _block_size_conv3x3(planes, planes, n_blocks, alpha=block_size_alpha)

        self.conv1 = lrm3x3(inplanes, planes, n_blocks, block_size_conv1, stride=stride, cache_attn=cache_attn)
        self.bn1 = norm_layer(planes)
        self.conv2 = lrm3x3(planes, planes, n_blocks, block_size_conv, cache_attn=cache_attn)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

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


# Modified from BasicBlock - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
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


# Modified from ResNet - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class LRMResNetV1(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 n_blocks=1, block_size_alpha=1, route_by_task=False):
        super(LRMResNetV1, self).__init__()
        self.task_id = None
        self.route_by_task = route_by_task
        self.n_blocks = n_blocks
        self.block_size_alpha = block_size_alpha

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = LRMConvV1(3, self.inplanes, (7, 7), n_blocks,
                               _block_size_conv(3, self.inplanes, 7, 7, n_blocks, alpha=block_size_alpha),
                               stride=2, padding=3)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        def _set_route_by_task(m):
            if type(m) == LRMConvV1:
                m.route_by_task = route_by_task

        # set route_by_task for all LRMConv modules
        self.apply(_set_route_by_task)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            block_size_downsample = _block_size(self.inplanes, planes * block.expansion, self.n_blocks,
                                                alpha=self.block_size_alpha)
            downsample = nn.Sequential(
                lrm1x1(self.inplanes, planes * block.expansion, self.n_blocks, block_size_downsample,
                       stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            n_blocks=self.n_blocks, block_size_alpha=self.block_size_alpha))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                n_blocks=self.n_blocks, block_size_alpha=self.block_size_alpha))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def set_task_id(self, task_id):
        """Set task id for the current task. Used for routing by task"""
        self.task_id = task_id

        def _set_task_id(m):
            if type(m) == LRMConvV1:
                m.task_id = task_id

        # set task_id for all LRMConv modules
        self.apply(_set_task_id)


def _lrm_resnet(Version, block, layers, num_classes=100, seed=1, disable_bn_stats=False, **kwargs):
    torch.manual_seed(seed)
    model = Version(block, layers, **kwargs)
    set_classification_layer(model, num_classes=num_classes)

    if disable_bn_stats:
        disable_bn_stats_tracking(model)
    return model


def lrm_resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _lrm_resnet(LRMResNetV1, LRMBlockV1, [2, 2, 2, 2], **kwargs)


archs = {
    'resnet18': resnet18,
    'lrm_resnet18': lrm_resnet18
}


# test model construction
if __name__ == '__main__':
    lrm_resnet18()
    resnet18()

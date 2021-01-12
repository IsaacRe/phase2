import torch.nn as nn
import torch.nn.functional as F
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


def lrm3x3(in_planes, out_planes, n_blocks, block_size, stride=1, cache_attn=True):
    """3x3 low-rank mixture convolution with padding"""
    return LRMConvV2(in_planes, out_planes, 3, n_blocks, block_size,
                     stride=stride, padding=1, cache_attn=cache_attn)


def lrm1x1(in_planes, out_planes, n_blocks, block_size, stride=1, cache_attn=True):
    """1x1 low-rank mixture convolution with padding"""
    return LRMConvV2(in_planes, out_planes, 1, n_blocks, block_size,
                     stride=stride, padding=0, cache_attn=cache_attn)


# Modified from BasicBlock - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class LRMBlockV2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 n_blocks=1, block_size_alpha=1.0, cache_attn=True):
        super(LRMBlockV2, self).__init__()

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
class LRMResNetV2(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 n_blocks=1, block_size_alpha=1, route_by='task-dynamic', fit_keys=False):
        super(LRMResNetV2, self).__init__()
        self.task_id = None
        self.route_by = route_by
        self.n_blocks = n_blocks
        self.block_size_alpha = block_size_alpha
        self.param_n_blocks = nn.Parameter(torch.LongTensor([n_blocks]), requires_grad=False)
        self.param_block_size_alpha = nn.Parameter(torch.FloatTensor([block_size_alpha]), requires_grad=False)

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
        self.conv1 = LRMConvV2(3, self.inplanes, (7, 7), n_blocks,
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

        def _set_route_by(m):
            if type(m) == LRMConvV2:
                m.route_by = route_by

        # set route_by for all LRMConv modules
        self.apply(_set_route_by)

        # running statistics
        self.running_entropy = {n: [] for n, m in self.named_modules() if type(m) == LRMConvV2}
        self.running_cls_route_div = {n: [] for n, m in self.named_modules() if type(m) == LRMConvV2}

        # disable grad on LRM block factors if we are only fitting keys
        if fit_keys:
            self.disable_block_grad()

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

        # accumulate LRM stats
        if self.route_by != 'task':
            self.accumulate_entropy()
            if self.task_id is not None:
                self.accumulate_class_routing_divergence()

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
            if type(m) == LRMConvV2:
                m.task_id = task_id

        # set task_id for all LRMConv modules
        self.apply(_set_task_id)

    def restructure_blocks(self, n_blocks, block_size_alpha):
        def restructure_lrm_conv(m):
            if type(m) == LRMConvV2:
                block_size = _block_size_conv(m.in_channels, m.out_channels, *m.kernel_size, n_blocks, block_size_alpha)
                m.restructure_blocks(n_blocks, block_size)

        self.apply(restructure_lrm_conv)

    def get_sims(self):
        sims = {}
        for n, m in self.named_modules():
            if type(m) == LRMConvV2:
                sims[n] = m.cached_attn
        return sims

    def accumulate_entropy(self):
        for n, m in self.named_modules():
            if type(m) == LRMConvV2:
                self.running_entropy[n] += [m.avg_entropy]

    def reset_entropy(self):
        self.running_entropy = {n: [] for n, m in self.named_modules() if type(m) == LRMConvV2}

    def get_entropy(self):
        return self.running_entropy

    def accumulate_class_routing_divergence(self, task_id=None):
        if task_id is None:
            task_id = self.task_id
        assert task_id is not None, 'task id is not set'
        for n, m in self.named_modules():
            if type(m) == LRMConvV2 and m.cached_attn is not None:
                sims = m.cached_attn.transpose(1, 3).reshape(-1, self.n_blocks)
                dim1 = sims.shape[0]
                self.running_cls_route_div[n] += [F.cross_entropy(sims, torch.LongTensor([task_id] * dim1)).item()]

    def reset_class_routing_divergence(self):
        self.running_cls_route_div = {n: [] for n, m in self.named_modules() if type(m) == LRMConvV2}

    def get_class_routing_divergence(self):
        return self.running_cls_route_div

    def reset_running_stats(self):
        self.reset_entropy()
        self.reset_class_routing_divergence()

    def disable_block_grad(self):
        def _disable_block_grad(m):
            if type(m) == LRMConvV2:
                m.disable_block_grad()
        self.apply(_disable_block_grad)

    def enable_block_grad(self):
        def _enable_block_grad(m):
            if type(m) == LRMConvV2:
                m.enable_block_grad()
        self.apply(_enable_block_grad)


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
    return _lrm_resnet(LRMResNetV2, LRMBlockV2, [2, 2, 2, 2], **kwargs)


archs = {
    'resnet18': resnet18,
    'lrm_resnet18': lrm_resnet18
}


# test model construction
if __name__ == '__main__':
    lrm_resnet18()
    resnet18()

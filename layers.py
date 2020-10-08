import torch.nn as nn
import torch


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()
        self.null = None

    def forward(self, x):
        if self.null is None:
            self.null = torch.zeros_like(x).to(x.device)
            self.null.requires_grad = False
        return self.null


class Pass(nn.Module):
    def __init__(self, detach=False):
        super(Pass, self).__init__()
        self.detach = detach

    def forward(self, x):
        if self.detach:
            x = x.detach()
        return x


class HashLinear(nn.Linear):
    def __init__(self, in_features, out_features, n_keys, optim='gd', detach=False, cache_attn=False, **kwargs):
        super(HashLinear, self).__init__(in_features, out_features, **kwargs)
        self.detach = detach
        self.do_cache = cache_attn
        self.cached_attn = None
        self.n_keys = n_keys

        self.key_op = nn.Linear(in_features, n_keys, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.val_op = nn.Linear(n_keys, out_features, bias=False)

    @property
    def hash_keys(self):
        return self.key_op.weight.data

    @property
    def hash_vals(self):
        return self.val_op.weight.data

    def cache_attn(self, cache=True):
        self.do_cache = cache

    def forward(self, x):
        if self.detach:
            x = x.detach()
        sims = self.key_op(x)  # [batch_size x n_keys]
        attn = self.softmax(sims)
        if self.cache_attn:
            self.cached_attn = attn.data.cpu()
        out = self.val_op(attn)  # [batch_size x out_features]
        return out


class HashConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_keys,
                 optim='gd', detach=False, cache_attn=False, padding=(1, 1), stride=(1, 1), t=1.0, hard=False):
        super(HashConv, self).__init__()
        self.hard = hard
        self.sm_temperature = t
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.detach = detach
        self.do_cache = cache_attn
        self.cached_attn = None
        self.avg_entropy = None
        self.n_keys = n_keys

        self.key_op = nn.Conv2d(in_channels, n_keys, kernel_size, padding=padding, stride=stride, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.val_op = nn.Conv2d(n_keys, out_channels, (1, 1), bias=False)

    @property
    def hash_keys(self):
        return self.key_op.weight.data

    @property
    def hash_vals(self):
        return self.val_op.weight.data

    def cache_attn(self, cache=True):
        self.do_cache = cache

    def set_temperature(self, t):
        self.sm_temperature = t

    def decay_temperature(self, decay):
        self.sm_temperature *= (1 - decay)

    def compute_attn_entropy(self, x):
        average_entropy = -(x * torch.log(x)).sum(dim=1).mean()
        self.avg_entropy = average_entropy

    def forward(self, x):
        if self.detach:
            x = x.detach()
        sims = self.key_op(x)  # [batch_size x n_keys x height x width]

        # if hard is true and we are evaluating, compute hard max operation instead of softmax
        if (not self.training) and self.hard:
            attn = torch.zeros_like(sims).to(sims.device)
            attn[sims == sims.max(dim=1)] = 1.0
        else:
            attn = self.softmax(sims / self.sm_temperature)

        self.compute_attn_entropy(attn)

        if self.do_cache:
            self.cached_attn = attn.data.cpu()

        out = self.val_op(attn)  # [batch_size x out_channels x height x width]

        return out


class LRMLinearV1(nn.Linear):
    def __init__(self, in_features, out_features, n_blocks, block_size, optim='gd', detach=False,
                 cache_attn=False, **kwargs):
        super(LRMLinearV1, self).__init__(in_features, out_features, **kwargs)
        self.detach = detach
        self.do_cache = cache_attn
        self.cached_attn = None
        self.n_blocks = n_blocks
        self.block_size = block_size

        self.key_op = nn.Linear(in_features, n_blocks, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.V_op = nn.Linear(in_features, n_blocks * block_size, bias=False)
        self.U_op = nn.Linear(n_blocks * block_size, out_features, bias=False)

    @property
    def hash_keys(self):
        return self.key_op.weight.data

    @property
    def V(self):
        return self.V_op.weight.data

    @property
    def U(self):
        return self.U_op.weight.data

    def cache_attn(self, cache=True):
        self.do_cache = cache

    def forward(self, x):
        if self.detach:
            x = x.detach()
        sims = self.key_op(x)           # [batch_size x n_blocks]
        attn = self.softmax(sims)
        if self.cache_attn:
            self.cached_attn = attn.data.cpu()
        out = self.V_op(x)              # [batch_size x n_blocks * block_size]
        out = out * attn[:, :, None].repeat(1, 1, self.block_size).reshape(-1, self.n_blocks * self.block_size)  # naive
        out = self.U_op(out)            # [batch_size x out_features]
        return out


class LRMConvV1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks, block_size,
                 optim='gd', detach=False, cache_attn=False, padding=(1, 1), stride=(1, 1), t=1.0, hard=False):
        super(LRMConvV1, self).__init__()
        self.hard = hard
        self.sm_temperature = t
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.detach = detach
        self.do_cache = cache_attn
        self.cached_attn = None
        self.avg_entropy = None
        self.n_blocks = n_blocks
        self.block_size = block_size

        self.key_op = nn.Conv2d(in_channels, n_blocks, kernel_size, padding=padding, stride=stride, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.V_op = nn.Conv2d(in_channels, n_blocks * block_size, kernel_size,
                              padding=padding, stride=stride, bias=False)
        self.U_op = nn.Conv2d(n_blocks * block_size, out_channels, (1, 1), bias=False)

    @property
    def hash_keys(self):
        return self.key_op.weight.data

    @property
    def hash_vals(self):
        return self.val_op.weight.data

    def cache_attn(self, cache=True):
        self.do_cache = cache

    def set_temperature(self, t):
        self.sm_temperature = t

    def decay_temperature(self, decay):
        self.sm_temperature *= (1 - decay)

    def compute_attn_entropy(self, x):
        average_entropy = -(x * torch.log(x)).sum(dim=1).mean()
        self.avg_entropy = average_entropy

    def forward(self, x):
        if self.detach:
            x = x.detach()
        sims = self.key_op(x)  # [batch_size x n_blocks x height x width]

        # if hard is true and we are evaluating, compute hard max operation instead of softmax
        if (not self.training) and self.hard:
            attn = torch.zeros_like(sims).to(sims.device)
            attn[sims == sims.max(dim=1)] = 1.0
        else:
            attn = self.softmax(sims / self.sm_temperature)

        self.compute_attn_entropy(attn)

        if self.do_cache:
            self.cached_attn = attn.data.cpu()

        h, w = attn.shape[2:]  # output featuremap height and width
        attn = attn[:, :, None].repeat(1, 1, self.block_size, 1, 1).reshape(-1, self.n_blocks * self.block_size, h, w)  # naive
                                    # [batch_size x n_blocks * block_size x height x width]

        out = self.V_op(x)
        out = out * attn   # [batch_size x n_blocks * block_size x height x width]
        out = self.U_op(out)        # [batch_size x out_channels x height x width]

        return out


# Modified BasicBlock used by ResNet18 - https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class InjectBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 inject_conv="hash", remove_bn2=True, detach=True, n_keys=256, block_size=64, cache_attn=False, temperature=1.0,
                 cut_residual=False):
        super(InjectBlock, self).__init__()

        ###################################
        remove_bn2 = remove_bn2 and (inject_conv is not None)
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
        self.inject_conv = inject_conv
        if inject_conv == "pass":
            self.conv2 = Pass(detach=detach)
        elif inject_conv == "hash":
            self.conv2 = HashConv(planes, planes, (3, 3), n_keys,
                                  detach=detach, cache_attn=cache_attn, t=temperature)
        elif inject_conv == "block":
            self.conv2 = BlockConv(planes, planes, (3, 3), n_keys, block_size,
                                   detach=detach, cache_attn=cache_attn, t=temperature)
        else:
            self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)

        if remove_bn2:
            self.bn2 = Pass()
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

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        ###################################
        # cut the residual connection if specified
        if not self.cut_residual:
            out += identity

        if self.inject_conv == "null":
            out = identity
        ###################################

        out = self.relu(out)

        return out

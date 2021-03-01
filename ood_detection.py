import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from model import resnet18
from experiment_utils.train_models import get_dataloaders_incr
from experiment_utils.utils.helpers import find_network_modules_by_name, set_torchvision_network_module
from args import *
from network_consolidation import ExperimentArgs


class OODArgs(ExperimentArgs):
    ARGS = {

    }


def main():
    data_args, experiment_args, model_args = parse_args(IncrDataArgs, ExperimentArgs, AllModelArgs)
    main_ood_detection(data_args, experiment_args, model_args)


def main_ood_detection(data_args, experiment_args, model_args):
    assert model_args.load_state_path, 'please specify a path to a pretrained model'
    state = torch.load(model_args.load_state_path)
    net = resnet18(num_classes=data_args.num_classes, seed=data_args.seed, disable_bn_stats=model_args.disable_bn_stats)
    if data_args.num_classes != state['fc.weight'].shape[0]:
        net.fc = nn.Linear(net.fc.in_features, state['fc.bias'].shape[0], bias=True)
    net.load_state_dict(state)
    net.cuda()

    train_loaders, val_loaders, test_loaders = get_dataloaders_incr(data_args, load_test=True)

    reinit_layers = find_network_modules_by_name(net, experiment_args.layer)
    layer_path = 'models/consolidation_experiments/incr_task/%d-layer' % len(reinit_layers)

    """net.bn1 = nn.BatchNorm2d(net.bn1.num_features, affine=False).cuda()
    net.layer1[0].bn1 = nn.BatchNorm2d(net.layer1[0].bn1.num_features, affine=False).cuda()
    net.layer1[0].bn2 = nn.BatchNorm2d(net.layer1[0].bn2.num_features, affine=False).cuda()
    net.layer1[1].bn1 = nn.BatchNorm2d(net.layer1[1].bn1.num_features, affine=False).cuda()
    net.layer1[1].bn2 = nn.BatchNorm2d(net.layer1[1].bn2.num_features, affine=False).cuda()
    net.layer2[0].bn1 = nn.BatchNorm2d(net.layer2[0].bn1.num_features, affine=False).cuda()
    net.layer2[0].bn2 = nn.BatchNorm2d(net.layer2[0].bn2.num_features, affine=False).cuda()
    net.layer2[0].downsample[1] = nn.BatchNorm2d(net.layer2[0].downsample[1].num_features, affine=False).cuda()
    net.layer2[1].bn1 = nn.BatchNorm2d(net.layer2[1].bn1.num_features, affine=False).cuda()
    net.layer2[1].bn2 = nn.BatchNorm2d(net.layer2[1].bn2.num_features, affine=False).cuda()
    net.layer3[0].bn1 = nn.BatchNorm2d(net.layer3[0].bn1.num_features, affine=False).cuda()
    net.layer3[0].bn2 = nn.BatchNorm2d(net.layer3[0].bn2.num_features, affine=False).cuda()
    net.layer3[0].downsample[1] = nn.BatchNorm2d(net.layer3[0].downsample[1].num_features, affine=False).cuda()
    net.layer3[1].bn1 = nn.BatchNorm2d(net.layer3[1].bn1.num_features, affine=False).cuda()
    net.layer3[1].bn2 = nn.BatchNorm2d(net.layer3[1].bn2.num_features, affine=False).cuda()"""
    net.layer4[0].bn1 = nn.BatchNorm2d(net.layer4[0].bn1.num_features, affine=False).cuda()
    net.layer4[0].bn2 = nn.BatchNorm2d(net.layer4[0].bn2.num_features, affine=False).cuda()
    net.layer4[0].downsample[1] = nn.BatchNorm2d(net.layer4[0].downsample[1].num_features, affine=False).cuda()
    net.layer4[1].bn1 = nn.BatchNorm2d(net.layer4[1].bn1.num_features, affine=False).cuda()
    net.layer4[1].bn2 = nn.BatchNorm2d(net.layer4[1].bn2.num_features, affine=False).cuda()

    def build_ood_conv(conv):
        in_ch = conv.in_channels // experiment_args.redundant_groups * experiment_args.redundant_groups
        out_ch = conv.out_channels // experiment_args.redundant_groups * experiment_args.redundant_groups
        return OODConv(in_ch, out_ch, conv.kernel_size, bias=conv.bias is not None,
                       stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                       groups=conv.groups)

    def load_weight(conv, file_path):
        state_dict = torch.load(file_path)
        conv.weight.data[:] = state_dict['weight']
        if 'bias' in state_dict.keys():
            conv.bias.data[:] = state_dict['bias']

    discriminator_params = []
    phi_params = []
    for i, layer_name in enumerate(experiment_args.layer):
        old_conv = reinit_layers[i]
        sup_conv = build_ood_conv(old_conv)
        set_torchvision_network_module(net, layer_name, sup_conv)
        load_weight(sup_conv, '%s/%s-task_0.pth' % (layer_path, layer_name))
        sup_conv.cuda()
        reinit_layers[i] = sup_conv
        discriminator_params += list(sup_conv.discriminator.parameters())
        phi_params += list(sup_conv.phi.parameters())

    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(discriminator_params + phi_params,
                                          lr=experiment_args.lr,
                                          nesterov=experiment_args.nesterov,
                                          momentum=experiment_args.momentum,
                                          weight_decay=experiment_args.weight_decay)
    discriminator_optim = torch.optim.SGD(discriminator_params,
                                          lr=experiment_args.lr,
                                          nesterov=experiment_args.nesterov,
                                          momentum=experiment_args.momentum,
                                          weight_decay=experiment_args.weight_decay)

    phi_optim = torch.optim.SGD(phi_params,
                                lr=experiment_args.lr,
                                nesterov=experiment_args.nesterov,
                                momentum=experiment_args.momentum,
                                weight_decay=experiment_args.weight_decay)

    def zero_grad():
        discriminator_optim.zero_grad()
        phi_optim.zero_grad()

    def discriminator_loss(m, length_pos):
        # get discriminator outputs for un-perturbed pos and neg inputs
        pos_logits = m.log_real[:length_pos]
        neg_logits = m.log_real[length_pos:]

        # get discriminator outputs for perturbed (pseudo pos) inputs
        pseudo_pos_logits = m.log_fake[length_pos:]

        total_pos = pos_logits.numel()
        total_neg = neg_logits.numel() + pseudo_pos_logits.numel()

        loss_pos = bce(pos_logits, torch.ones_like(pos_logits))
        loss_neg = bce(neg_logits, torch.zeros_like(neg_logits))
        #loss_pseudo_pos = bce(pseudo_pos_logits, torch.zeros_like(pseudo_pos_logits))
        #loss_all_neg = loss_neg + loss_pseudo_pos

        # weight positive and negative samples evenly
        #loss = (loss_pos * total_neg + loss_all_neg * total_pos) / 2 / total_neg / total_pos
        loss = (loss_pos + loss_neg) / 2

        pred_real = torch.sigmoid(m.log_real).round()
        acc_real = pred_real[:length_pos].sum().item() - (pred_real[length_pos:] - 1).sum().item()
        acc_real /= pred_real.numel()

        pred_fake = torch.sigmoid(pseudo_pos_logits).round()
        acc_fake = -(pred_fake - 1).sum().item() / pred_fake.numel()

        return loss, acc_real, acc_fake

    def phi_loss(m, length_pos):
        # get discriminator outputs for perturbed (pseudo pos) inputs
        pseudo_pos_logits = m.log_fake[length_pos:]

        loss = bce(pseudo_pos_logits, torch.ones_like(pseudo_pos_logits))

        pred = torch.sigmoid(pseudo_pos_logits).round()
        acc = -(pred - 1).sum().item() / pred.numel()

        # TODO impose regularizations on transformed output
        return loss, acc

    discriminator_losses_by_layer = {n: [] for n in experiment_args.layer}
    phi_losses_by_layer = {n: [] for n in experiment_args.layer}
    real_accs_by_layer = {n: [] for n in experiment_args.layer}
    fake_accs_by_layer = {n: [] for n in experiment_args.layer}
    optimize = 'discriminator'
    pbar = tqdm(total=min(map(lambda x: len(x), train_loaders[:2])))
    for (_, x0, y0), (_, x1, y1) in zip(*train_loaders[:2]):
        x0, x1 = x0.to(0), x1.to(0)
        length_pos = len(y0)
        y = torch.zeros(len(y0) + len(y1)).to(0)
        y[:length_pos] = 1
        x = torch.cat([x0, x1], dim=0)
        net(x)

        # discriminator update
        if optimize == 'discriminator':
            for n, m in net.named_modules():
                if type(m) == OODConv:
                    """#loss = bce(m.log_real, y[:,None,None,None].repeat(1, *m.log_real.shape[1:]))
                    loss = bce(m.log_real[:length_pos], torch.ones_like(m.log_real[:length_pos]))
                    loss += bce(m.log_real[length_pos:], torch.zeros_like(m.log_real[length_pos:]))
                    loss /= 2
                    discriminator_losses_by_layer[n] += [loss.item()]
                    loss.backward()"""

                    loss, acc_real, acc_fake = discriminator_loss(m, length_pos)

                    discriminator_losses_by_layer[n] += [loss.item()]

                    real_accs_by_layer[n] += [acc_real]
                    fake_accs_by_layer[n] += [acc_fake]

                    loss.backward()
            discriminator_optim.step()
        # phi update
        elif optimize == 'phi':
            for n, m in net.named_modules():
                if type(m) == OODConv:
                    loss, acc_fake = phi_loss(m, length_pos)

                    phi_losses_by_layer[n] += [loss.item()]

                    real_accs_by_layer[n] += [real_accs_by_layer[n][-1]]
                    fake_accs_by_layer[n] += [acc_fake]

                    loss.backward()
            phi_optim.step()

        zero_grad()
        pbar.update(1)
    pass




class OODConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(OODConv, self).__init__(*args, **kwargs)
        self.log_real = None  # logits for real samples
        self.log_fake = None  # logits for fake samples
        self.phi_out = None  # faked in-domain data
        self.discriminator = nn.Conv2d(self.in_channels, 1, (1, 1), bias=True)
        self.phi = nn.Conv2d(self.in_channels, self.in_channels, (1, 1), bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.phi_out = self.phi(input.detach())
        self.log_real = self.discriminator(input.detach())
        self.log_fake = self.discriminator(self.phi_out)

        return super(OODConv, self).forward(input)


if __name__ == '__main__':
    main()
from os import mkdir
from os.path import exists
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from experiment_utils.utils.layer_injection import LayerInjector
from args import *
from experiment_utils.train_models import test, train, get_dataloaders_incr, get_dataloaders, get_subset_data_loaders
from experiment_utils.utils.helpers import find_network_modules_by_name, set_torchvision_network_module
from model import resnet18


def main():
    data_args, experiment_args, model_args = parse_args(IncrDataArgs, ExperimentArgs, AllModelArgs)
    if experiment_args.experiment == 'consolidate':
        main_consolidate(data_args, experiment_args, model_args)
    else:
        main_dropout(data_args, experiment_args, model_args)


def drop_output_nodes(state, keep_outputs):
    weight = state['fc.weight']
    state['fc.weight'] = weight[keep_outputs, :]

    if 'fc.bias' in state.keys():
        bias = state['fc.bias']
        state['fc.bias'] = bias[keep_outputs]


def format_tuple_arg(t_or_not):
    if hasattr(t_or_not, '__iter__'):
        list_ = []
        for elem in t_or_not:
            assert type(elem) == int, 'data type not understood'
            list_ += [elem]
        return tuple(list_)
    assert type(t_or_not) == int, 'data type not understood'
    return tuple([t_or_not] * 2)


def main_dropout(data_args, train_args, model_args):
    train_loader, val_loader, test_loader = get_dataloaders(data_args, load_test=False)
    #train_loader, val_loader, test_loader = get_dataloaders_incr(data_args, load_test=False)
    #train_loader, val_loader = train_loader[0], val_loader[0]

    assert model_args.load_state_path, 'please specify a path to a pretrained model'
    state = torch.load(model_args.load_state_path)
    net = resnet18(num_classes=data_args.num_classes, seed=data_args.seed, disable_bn_stats=model_args.disable_bn_stats)
    net.load_state_dict(state)
    net.cuda()

    drop_features(train_args, net, train_loader, val_loader, device=0)


def main_consolidate(data_args, train_args, model_args):
    assert model_args.load_state_path, 'please specify a path to a pretrained model'
    state = torch.load(model_args.load_state_path)
    net = resnet18(num_classes=data_args.num_classes, seed=data_args.seed, disable_bn_stats=model_args.disable_bn_stats)
    if data_args.num_classes != state['fc.weight'].shape[0]:
        net.fc = nn.Linear(net.fc.in_features, state['fc.bias'].shape[0], bias=True)
    net.load_state_dict(state)
    net.cuda()

    if train_args.single_task:
        consolidate_single_task(data_args, train_args, net, device=0)
    else:
        consolidate_multi_task(data_args, train_args, net, device=0)


def apply_module_method_if_exists(model, method_name):
    def model_apply_fn(*args, **kwargs):
        def module_apply_fn(m):
            if hasattr(m, method_name) and m != model:
                getattr(m, method_name)(*args, **kwargs)

        model.apply(module_apply_fn)

    return model_apply_fn


def consolidate_multi_task(data_args, train_args, model, device=0):
    train_loaders, val_loaders, test_loaders = get_dataloaders_incr(data_args, load_test=True)
    _, _, test_ldr = get_dataloaders(data_args, load_train=False)

    reinit_layers = find_network_modules_by_name(model, train_args.layer)

    if train_args.superimpose:

        # define SuperConv model-wise apply methods
        model.superimpose = apply_module_method_if_exists(model, 'superimpose')
        model.load_superimposed_weight = apply_module_method_if_exists(model, 'load_superimposed_weight')
        model.update_component = apply_module_method_if_exists(model, 'update_component')
        model.scale_supconv_grads = apply_module_method_if_exists(model, 'scale_grad')

        def build_super_conv(conv):
            in_ch = conv.in_channels // train_args.redundant_groups * train_args.redundant_groups
            out_ch = conv.out_channels // train_args.redundant_groups * train_args.redundant_groups
            return SuperConv(in_ch, out_ch, conv.kernel_size, bias=conv.bias is not None,
                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                             groups=conv.groups * train_args.redundant_groups, drop_groups=train_args.drop_groups,
                             weight_sup=train_args.weight_sup_method == 'avg')

        for i, layer_name in enumerate(train_args.layer):
            old_conv = reinit_layers[i]
            sup_conv = build_super_conv(old_conv)
            set_torchvision_network_module(model, layer_name, sup_conv)
            sup_conv.cuda()
            reinit_layers[i] = sup_conv

    elif train_args.l2:
        model.update_previous_params = apply_module_method_if_exists(model, 'update_previous_weight')

        def build_l2_conv(conv):
            in_ch = conv.in_channels // train_args.redundant_groups * train_args.redundant_groups
            out_ch = conv.out_channels // train_args.redundant_groups * train_args.redundant_groups
            return L2Conv(in_ch, out_ch, conv.kernel_size, bias=conv.bias is not None,
                          stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                          groups=conv.groups * train_args.redundant_groups)

        for i, layer_name in enumerate(train_args.layer):
            old_conv = reinit_layers[i]
            l2_conv = build_l2_conv(old_conv)
            set_torchvision_network_module(model, layer_name, l2_conv)
            l2_conv.cuda()
            reinit_layers[i] = l2_conv

    # disable affine and running stats of retrained bn layers
    """model.bn1 = nn.BatchNorm2d(model.bn1.num_features, affine=False).cuda()
    model.layer1[0].bn1 = nn.BatchNorm2d(model.layer1[0].bn1.num_features, affine=False).cuda()
    model.layer1[0].bn2 = nn.BatchNorm2d(model.layer1[0].bn2.num_features, affine=False).cuda()
    model.layer1[1].bn1 = nn.BatchNorm2d(model.layer1[1].bn1.num_features, affine=False).cuda()
    model.layer1[1].bn2 = nn.BatchNorm2d(model.layer1[1].bn2.num_features, affine=False).cuda()
    model.layer2[0].bn1 = nn.BatchNorm2d(model.layer2[0].bn1.num_features, affine=False).cuda()
    model.layer2[0].bn2 = nn.BatchNorm2d(model.layer2[0].bn2.num_features, affine=False).cuda()
    model.layer2[0].downsample[1] = nn.BatchNorm2d(model.layer2[0].downsample[1].num_features, affine=False).cuda()
    model.layer2[1].bn1 = nn.BatchNorm2d(model.layer2[1].bn1.num_features, affine=False).cuda()
    model.layer2[1].bn2 = nn.BatchNorm2d(model.layer2[1].bn2.num_features, affine=False).cuda()
    model.layer3[0].bn1 = nn.BatchNorm2d(model.layer3[0].bn1.num_features, affine=False).cuda()
    model.layer3[0].bn2 = nn.BatchNorm2d(model.layer3[0].bn2.num_features, affine=False).cuda()
    model.layer3[0].downsample[1] = nn.BatchNorm2d(model.layer3[0].downsample[1].num_features, affine=False).cuda()
    model.layer3[1].bn1 = nn.BatchNorm2d(model.layer3[1].bn1.num_features, affine=False).cuda()
    model.layer3[1].bn2 = nn.BatchNorm2d(model.layer3[1].bn2.num_features, affine=False).cuda()"""
    model.layer4[0].bn1 = nn.BatchNorm2d(model.layer4[0].bn1.num_features, affine=False).cuda()
    model.layer4[0].bn2 = nn.BatchNorm2d(model.layer4[0].bn2.num_features, affine=False).cuda()
    model.layer4[0].downsample[1] = nn.BatchNorm2d(model.layer4[0].downsample[1].num_features, affine=False).cuda()
    model.layer4[1].bn1 = nn.BatchNorm2d(model.layer4[1].bn1.num_features, affine=False).cuda()
    model.layer4[1].bn2 = nn.BatchNorm2d(model.layer4[1].bn2.num_features, affine=False).cuda()

    model.eval()
    # if not updating bn layer during training, disable model's train mode
    if not train_args.fit_bn_stats:
        model.train = lambda *args, **kwargs: None

    # test pretrained model accuracy
    """pt_accuracies = []
    for i, test_loader in enumerate(test_loaders):
        c, t = test(model, test_loader, device=device, multihead=True)
        acc = (c.sum() / t.sum()).item()
        print('Pretrained model accuracy for task %d: %.2f' % (i, acc * 100.))
        pt_accuracies += [acc]"""

    def save_layer(save_path, suffix='.pth'):
        for layer, name in zip(reinit_layers, train_args.layer):
            layer.cpu()
            torch.save(layer.state_dict(), save_path + name + suffix)
            layer.cuda()

    def load_layer(load_path, suffix='.pth'):
        for layer, name in zip(reinit_layers, train_args.layer):
            layer.cpu()
            layer.load_state_dict(torch.load(load_path + name + suffix))
            layer.cuda()

    base_dir = 'models/consolidation_experiments/%s/' % train_args.experiment_id
    base_path = base_dir + '%d-layer/' % len(train_args.layer)

    if not exists(base_dir):
        mkdir(base_dir)

    if not exists(base_path):
        mkdir(base_path)

    # covariance experimentation
    """from sklearn.covariance import EmpiricalCovariance

    def get_cov(ldr, sample_idxs=slice(0, 64), normalize=False):
        feature_layer = reinit_layers[-1]
        load_layer(base_path, suffix='-task_0.pth')
        f1 = compute_features(model, feature_layer, ldr)
        load_layer(base_path, suffix='-task_1.pth')
        f2 = compute_features(model, feature_layer, ldr)

        # subsample
        f1 = torch.cat(f1)[:,sample_idxs].flatten(start_dim=1)
        f2 = torch.cat(f2)[:,sample_idxs].flatten(start_dim=1)
        fcat = torch.cat([f1, f2], dim=1)

        length = f1.shape[1]

        cov = EmpiricalCovariance().fit(fcat).covariace_
        
        if normalize:
            cov = cov ** 2 / (cov ** 2).sum(axis=0)[None, :] / (cov ** 2).sum(axis=1)[:, None]
        
        cov1 = cov[:length, :length]
        cov2 = cov[length:, length:]
        xcov = cov[:length, length:]

        return cov1, cov2, xcov

    def get_kernel_sim():
        pass"""

    # save pretrained parameterization of the layer
    save_layer(base_path, suffix='-full.pth')

    # reinitialize the layer
    if not train_args.superimpose:
        for layer in reinit_layers:
            layer.reset_parameters()
        save_layer(base_path, suffix='-reinit.pth')

    accuracies = []
    # train separately on each subtask
    for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        train(train_args, model, train_loader, val_loader, device=device, optimize_modules=reinit_layers,
              multihead=True)
        if train_args.superimpose:
            model.superimpose(True)
        accs = []
        accuracies += [accs]
        for j, test_loader in enumerate(test_loaders):
            c, t = test(model, test_loader, device=device, multihead=True)
            acc = (c.sum() / t.sum()).item()
            accs += [acc]
            print('Task-%d-trained model accuracy for task %d: %.2f' % (i, j, acc * 100.))

        # load superimposed weight into memory to be saved
        if train_args.superimpose:
            model.load_superimposed_weight()

        # save trained layer
        save_layer(base_path, suffix='-task_%d.pth' % i)

        if not train_args.incremental:
            # reinitialize the layer
            load_layer(base_path, suffix='-reinit.pth')

        # reset weight and component in SuperConv
        if train_args.superimpose:
            model.update_component()

        # update previous parameterization if conducting l2 penalty
        elif train_args.l2:
            model.update_previous_params()

    # consolidate using kernel averaging
    if not train_args.incremental:
        print('Consolidating separately trained layers...')

    """threshold = 0.3
    for layer, name in zip(reinit_layers, train_args.layer):
        w = torch.load(base_path + '%s-task_%d.pth' % (name, 0))['weight']
        n_consolidated = torch.ones_like(w)
        for i in range(1, 5):
            new_w = torch.load(base_path + '%s-task_%d.pth' % (name, i))['weight']
            # TODO normalize by distribution of weights in each layer
            diff = ((w - new_w) ** 2).sum(axis=(1, 2, 3)) ** (1/2)
            consolidate = diff < threshold
            w[consolidate] = w[consolidate] + new_w[consolidate]
            n_consolidated[consolidate] += 1

        perc_consolidated = len(np.where(n_consolidated > 1)[0]) / n_consolidated.flatten().shape[0]
        print('%.2f %% of weights consolidated for layer %s' % (perc_consolidated * 100., name))

        w /= n_consolidated
        layer.cpu()
        layer.weight.data[:] = w
        layer.cuda()"""

    if not train_args.incremental:
        for layer, name in zip(reinit_layers, train_args.layer):
            w = 0
            for i in range(len(train_loaders)):
                w = w + torch.load(base_path + '%s-task_%d.pth' % (name, i))['weight']

            layer.weight.data[:] = w.to(device) / 5

    # test consolidated layer
    model.train()
    consolidated_accs = []
    for i, test_loader in enumerate(test_loaders):
        c, t = test(model, test_loader, device=device, multihead=True)
        acc = (c.sum() / t.sum()).item()
        print('Accuracy of consolidated model on task %d: %.2f' % (i, acc * 100.))
        consolidated_accs += [acc]


def consolidate_single_task(data_args, train_args, model, device=0):
    train_loaders, val_loader = get_subset_data_loaders(data_args, train_args.num_samples)

    reinit_layer, = find_network_modules_by_name(model, [train_args.layer])

    # test initial accuracy
    c, t = test(model, val_loader)
    pt_accuracy = (c.sum() / t.sum()).item()
    print('Accuracy of fully trained model: %.2f' % (pt_accuracy * 100.))

    def save_layer(save_path):
        reinit_layer.cpu()
        torch.save(reinit_layer.state_dict(), save_path)
        reinit_layer.cuda()

    def load_layer(load_path):
        reinit_layer.cpu()
        reinit_layer.load_state_dict(torch.load(load_path))
        reinit_layer.cuda()

    base_dir = 'models/consolidation_experiments/same_task/'
    base_path = base_dir + train_args.layer + '-diff_reinit-'

    # save pretrained parameterization of final layer
    save_layer(base_path + 'full.pth')

    # reinit final feature layer
    reinit_layer.reset_parameters()
    save_layer(base_path + 'reinit_0.pth')

    accuracies = []

    # train final layer separately on each subset of data
    for i, loader in enumerate(train_loaders):
        train(train_args, model, loader, val_loader, device=device, optimize_modules=[reinit_layer])
        c, t = test(model, val_loader)
        accuracies += [(c.sum() / t.sum()).item()]
        print('Accuracy of model trained on subset %d: %.2f' % (i, accuracies[-1] * 100.))

        save_layer(base_path + str(i) + '.pth')

        if not train_args.incremental:
            # use different reinitialization
            #load_layer(base_path + 'reinit.pth')
            reinit_layer.reset_parameters()
            save_layer(base_path + 'reinit_%d.pth' % (i + 1))

    if not train_args.incremental:
        # attempt to consolidate separately trained layers into a single representation
        print('Consolidating separately trained layers...')

        # 1 - naive averaging
        state1 = torch.load(base_path + '0.pth')
        state2 = torch.load(base_path + '1.pth')

        w = (state1['weight'] + state2['weight']) / 2
        reinit_layer.weight.data[:] = w.to(device)

    # test consolidated model
    c, t = test(model, val_loader)
    consolidated_acc = (c.sum() / t.sum()).item()
    print('Accuracy of consolidated model: %.2f' % (consolidated_acc * 100.))


def compute_features(model, layer, loader, device=0):
    from experiment_utils.utils.model_tracking import ModuleTracker, TrackingProtocol

    tracker = ModuleTracker(TrackingProtocol('out'), layer)

    features = []
    with torch.no_grad():
        for i, x, y in loader:
            x = x.to(device)
            with tracker.track():
                model(x)
                features += [tracker.gather_module_var(layer.name, 'out')]

    return features


def drop_features(args, model, train_loader, val_loader, device=0):
    cross_ent = nn.CrossEntropyLoss()
    layer_injector = LayerInjector(model)
    dropout_layer = AdaptiveDropout(args.num_features, suppress_func=args.suppress_func)
    dropout_layer.cuda()

    # get loss before feature suppression
    with torch.no_grad():
        print('Computing loss before feature suppression...', end=' ')
        _, x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        loss = cross_ent(model(x), y)
        print('train: %.4f,' % loss.item(), end=' ')

        _, x, y = next(iter(val_loader))
        x, y = x.to(device), y.to(device)
        loss = cross_ent(model(x), y)
        print('val: %.4f' % loss.item())

    layer_injector.insert_at_layer(dropout_layer, args.layer)
    net_optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                nesterov=args.nesterov, momentum=args.momentum)
    if args.drop_adam:
        dropout_optim = torch.optim.Adam(dropout_layer.parameters(), lr=args.drop_lr,
                                         weight_decay=args.drop_weight_decay)
    else:
        dropout_optim = torch.optim.SGD(dropout_layer.parameters(), lr=args.drop_lr,
                                        weight_decay=args.drop_weight_decay,
                                        nesterov=args.drop_nesterov, momentum=args.drop_momentum)

    # gradient increase as suppression increases uniformly
    """
    import numpy as np
    grads = []
    for i in np.arange(1, 0, -0.02):
        # using relu suppress func
        dropout_layer.weight.data[:] = i

        for (_, x, y) in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = cross_ent(out, y)
            loss.backward()

        grads += [dropout_layer.weight.grad.mean().item()]
        dropout_optim.zero_grad()

    np.savez('feature-suppression-grads.npz', grads=np.array(grads), signal_intensity=np.arange(1, 0, -0.02))

    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    def log_approx(data, a, b):
        return np.exp(-a * data) / b

    # discard last 6 values (anomaly when we approach full suppression)
    (a, b), _ = curve_fit(log_approx, np.arange(1, 0, -0.02)[:-6], -np.array(grads)[:-6])

    # a = 5.52240836
    # b = 1.48766782
    def plot(a, b):
        plt.plot(np.arange(1, 0, -0.02)[:-6], list(map(lambda x: log_approx(x, a, b), np.arange(1, 0, -0.02)))[:-6],
                 label='approx')
        plt.plot(np.arange(1, 0, -0.02)[:-6], -np.array(grads)[:-6], label='grads')
        plt.legend()
        plt.show()"""

    # accuracy decrease as features drop
    """
    accs = []
    class_correct = {}
    import numpy as np
    with torch.no_grad():
        for i in range(512):
            print('Testing for %d features retained' % i)
            dropout_layer.weight.data[:] = 100
            dropout_layer.weight.data[np.random.choice(512, i)] = 0
            c, t = test(model, val_loader)

            dropout_layer.weight.data[:] = 100
            dropout_layer.weight.data[np.random.choice(512, i)] = 0
            c, t = test(model, val_loader)

            class_correct[str(i)] = c
            accs += [c.sum() / t.sum() * 100.]

            class_correct['total'] = t
            class_correct['average'] = np.stack(accs)
            np.savez('drop_feature_acc-cifar10.npz', **class_correct)"""

    grads_l = []
    grads_s = []
    weights = []

    for e in range(args.epochs):

        for i, (_, x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = cross_ent(out, y)

            #assert not dropout_layer.weight.min().item() <= 0

            loss.backward(retain_graph=True)
            grad_l = dropout_layer.weight.grad.cpu()
            dropout_optim.zero_grad()

            dropout_layer.compute_loss().backward()
            grad_s = dropout_layer.weight.grad.cpu()
            dropout_optim.zero_grad()

            grads_s += [grad_s.mean().item()]
            grads_l += [grad_l.mean().item()]
            weights += [dropout_layer.weight.mean().item()]

            loss.backward()
            suppression_loss = dropout_layer.compute_loss()
            assert not np.isnan(suppression_loss.item())
            (suppression_loss * args.suppress_rate).backward()

            grad = dropout_layer.weight.grad.cpu()
            positive_features = len(np.where(grad > 0)[0]) / 512
            positive_mass = (grad[grad > 0].sum() / grad.abs().sum()).item()

            net_optim.step()
            dropout_optim.step()
            net_optim.zero_grad()
            dropout_optim.zero_grad()

            if i % 5 == 0:
                suppression = dropout_layer.compute_suppression().data.cpu()
                num_pruned = len(np.where(suppression <= 0)[0])
                print('Epoch %d, iter %d, loss: %.4f, total signal retention: %.4f / %d,' %
                      (e, i, loss.item(), suppression_loss.item(), args.num_features), end='')

                print(' total pruned: %d / %d, %% positive gradient features: %.2f, %% positive gradient mass %.2f' %
                      (num_pruned, args.num_features, positive_features * 100., positive_mass * 100.))
        test(model, val_loader, device=device, multihead=False)


def collect_l2_weight(model, loader, method='mas', device=0):
    total = 0
    for i, x, y in tqdm(loader):
        x = x.to(device)
        out = model(x)
        loss = (out ** 2).sum()
        loss.backward()
        total += len(y)

    # set regularization weights
    for n, m in model.named_modules():
        if hasattr(m, 'l2_weight'):
            if type(m.l2_weight) != torch.Tensor:
                m.l2_weight = 0
            m.l2_weight = m.l2_weight + m.weight.grad.abs() / total

    model.zero_grad()


class SuperConv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
                 stride=1, padding=1, dilation=1, groups=1,
                 normalize=False, drop_groups=False, weight_sup=False):
        self.weight_sup = weight_sup

        # set redundant_groups
        self.redundant_groups = groups > 1 and not drop_groups
        self.drop_groups = drop_groups
        groups = groups if drop_groups else 1
        super(SuperConv, self).__init__(in_channels, out_channels, kernel_size,
                                        bias=bias, stride=stride, padding=padding,
                                        dilation=dilation, groups=groups)

        # load random intialization onto cpu for later exposures
        self.random_init = self.weight.data.clone().cpu()

        # intialize component tensor to zero
        self.component = torch.zeros_like(self.weight)

        # loss from L2 penalty on weight movement
        self.l2 = None

        self.consolidated_weights = 0
        self.superimposing = True

        # for regularization weighting schemes (MAS, EWC, SI, etc.)
        self.l2_weight = 1

        # use cosine similarity as function output
        self.normalize = normalize

        self.drop_groups = drop_groups

    def _normalize_x(self, x):
        if self.groups > 1:
            raise NotImplementedError
        return F.conv2d(x ** 2, torch.ones(1, *self.weight.shape[1:]).to(self.weight.device), stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

    def _normalize_w(self, x, w=None):
        if self.groups > 1:
            raise NotImplementedError
        if w is None:
            w = self.weight
        return F.conv2d(torch.ones_like(x), w ** 2, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

    def normalize_weight(self):
        l2 = (self.weight.data ** 2).sum(dim=3).sum(dim=2).sum(dim=1)  # [out channels]
        self.weight.data[:] = self.weight.data / l2[:, None, None, None]

    def normalize_component(self):
        l2 = (self.component ** 2).sum(dim=3).sum(dim=2).sum(dim=1)  # [out channels]
        self.component[:] = self.component / l2[:, None, None, None]

    def _shuffle_groups(self):
        group_idx = np.random.choice(self.groups, self.groups, replace=False)
        group_idx = group_idx[:,None].repeat(self.out_channels // self.groups, 1).reshape(-1)
        self.weight.data[:] = self.weight.data[group_idx]

        group_idx = np.random.choice(self.groups, self.groups, replace=False)
        group_idx = group_idx[:, None].repeat(self.out_channels // self.groups, 1).reshape(-1)
        self.component[:] = self.component[group_idx]

    def _duplicate_mean(self, y):
        batch, out_ch, h, w = y.shape
        y = y.reshape(batch, self.groups, out_ch // self.groups, h, w)
        y = y.mean(dim=1)
        y = y[:,None].repeat(1, self.groups, 1, 1, 1).reshape(batch, out_ch, h, w)
        return y

    def _drop_groups(self, y):
        # set outputs for all but first group to zero
        y[:, self.out_channels // self.groups:] -= y[:, self.out_channels // self.groups:]
        return y

    def _superimpose(self):
        if self.component.device != self.weight.device:
            self.component = self.component.to(self.weight.device)

        # scale weight and component
        if self.weight_sup:
            sum_comp = self.weight + self.component * self.consolidated_weights * self.l2_weight
            return sum_comp / (1 + self.consolidated_weights * self.l2_weight)
        return (self.weight + self.component * self.consolidated_weights) / (1 + self.consolidated_weights)

    def superimpose(self, mode=True):
        self.superimposing = mode

    def train(self, mode=True):
        super(SuperConv, self).train(mode=mode)
        self.superimpose(mode=mode)

    def load_superimposed_weight(self):
        if self.component.device != self.weight.device:
            self.component = self.component.to(self.weight.device)
        self.weight.data[:] = self._superimpose().data
        if self.normalize:
            self.normalize_weight()
        self.component[:] = 0

        # update number of consolidated weights
        self.consolidated_weights += 1

    def update_component(self):
        random_init = self.random_init.to(self.weight.device)

        # get sum of all updates applied during this data exposure and store in component
        self.component = self.weight.data.clone()  # - random_init

        # reset weight to random init
        self.weight.data[:] = random_init

        if self.normalize:
            self.normalize_weight()
            self.normalize_component()

    def iterate(self):
        self.load_superimposed_weight()
        self.update_component()

    def scale_grad(self):
        self.weight.grad /= 2

    def compute_l2(self):
        if self.component.device != self.weight.device:
            self.component = self.component.to(self.weight.device)

        return ((self.weight - self.component) ** 2 * self.l2_weight).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shuffle groups
        if self.redundant_groups:
            self._shuffle_groups()

        if self.superimposing:
            weight = self._superimpose()
        else:
            weight = self.weight
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=self.groups)
        if self.normalize:
            x_l2 = self._normalize_x(x)
            w_l2 = self._normalize_w(x, w=weight)
            out /= w_l2 * x_l2

        if self.redundant_groups:
            out = self._duplicate_mean(out)
        if self.drop_groups or self.redundant_groups:
            out = self._drop_groups(out)

        return out


class L2Conv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
                 stride=1, padding=1, dilation=1, groups=1, normalize=False):
        super(L2Conv, self).__init__(in_channels, out_channels, kernel_size,
                                     bias=bias, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups)
        self.normalize = normalize
        self.previous_weight = torch.zeros_like(self.weight)

        # for regularization weighting schemes (MAS, EWC, SI, etc.)
        self.l2_weight = 1

    def _normalize_x(self, x):
        if self.groups > 1:
            raise NotImplementedError
        return F.conv2d(x ** 2, torch.ones(1, *self.weight.shape[1:]).to(self.weight.device), stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

    def _normalize_w(self, x):
        if self.groups > 1:
            raise NotImplementedError
        return F.conv2d(torch.ones_like(x), self.weight ** 2, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

    def normalize_weight(self):
        l2 = (self.weight.data ** 2).sum(dim=3).sum(dim=2).sum(dim=1)  # [out channels]
        self.weight.data[:] = self.weight.data / l2[:,None,None,None]

    def update_previous_weight(self):
        self.previous_weight = self.weight.data.clone()

    def compute_l2(self):
        if self.previous_weight.device != self.weight.device:
            self.previous_weight = self.previous_weight.to(self.weight.device)

        return ((self.weight - self.previous_weight) ** 2 * self.l2_weight).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super(L2Conv, self).forward(x)
        if self.normalize:
            x_l2 = self._normalize_x(x)
            w_l2 = self._normalize_w(x)
            out /= w_l2 * x_l2
        return out


class StitchingLayer(nn.Module):
    pass


class AdaptiveDropout(nn.Module):

    a = 5.522408355375256
    b = 0.18099237673904825  # 1.4876678179230516
    C = 1.0004864229292847

    def __init__(self, in_features, detach=True, suppress_func='normal'):
        super(AdaptiveDropout, self).__init__()
        self.suppress_func = suppress_func
        self.detach = detach
        self.in_features = in_features
        self.weight = nn.Parameter(torch.ones(in_features))

        # used for normal suppression function
        self.normal = Normal(0, 1)
        self.scale = 1 / 0.3989  # 1 over the pdf value at 0

    def forward(self, x):
        if self.detach:
            x = x.detach()
        weight = self.compute_suppression()
        if len(x.shape) == 2:
            return x * weight[None, :]
        else:
            ret = x * weight[None, :, None, None]
            assert not np.any(np.isnan(ret.data.cpu().numpy()))
            return ret

    def compute_suppression(self):
        if self.suppress_func == 'normal':
            unscaled = torch.exp(self.normal.log_prob(self.weight))
            return unscaled * self.scale
        elif self.suppress_func == 'relu':
            return nn.functional.relu(self.weight)
        elif self.suppress_func == 'exp':
            # using a and b from above experiment, integrating and solving for C = 1 + exp(-a) / (a * b) =
            ret = -torch.exp(-self.a * self.weight) / self.a / self.b + self.C
            ret = nn.functional.relu(ret)
            return ret
        else:
            return torch.exp(-torch.abs(self.weight))

    def compute_loss(self):
        return self.compute_suppression().sum()


class FlattenConv(nn.Module):

    def __init__(self, in_channel, kernel_size: Any = 3, stride: Any = 1, padding: Any = 1):
        super(FlattenConv, self).__init__()
        self.in_channel = in_channel
        self.kernel_size = format_tuple_arg(kernel_size)
        self.stride = format_tuple_arg(stride)
        self.padding = format_tuple_arg(padding)
        spatial_fan = self.kernel_size[0] * self.kernel_size[1]
        self.fan_out = in_channel * spatial_fan
        self.weight = torch.diag(torch.ones(spatial_fan)).repeat(in_channel, 1).reshape(self.fan_out,
                                                                                        1,
                                                                                        *self.kernel_size)

    def __repr__(self):
        return 'FlattenConv(in_channel=%d, kernel_size=%s, stride=%s, padding=%s)' % \
               (self.in_channel, str(self.kernel_size), str(self.stride), str(self.padding))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(input, self.weight, stride=self.stride, padding=self.padding,
                        groups=self.in_channel)


class BlockRoutingConv(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, kernel_size=3, stride=1, padding=1, route_forward=False):
        kernel_size = format_tuple_arg(kernel_size)
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        self.inputs_per_block = fan_in // blocks

        self.flatten = FlattenConv(in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.classifier = nn.Conv2d(in_channels=fan_in, out_channels=blocks, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(in_channels=fan_in, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.route_forward = route_forward
        self.logits = None
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        flat = self.flatten(input)  # [batch X in_channel * k_h * k_w X h X w]
        logits = self.classifier(flat)  # [batch X blocks X h X w]
        batch, blocks, h, w = logits.shape
        if self.training:
            self.logits = logits
        preds = self.sigmoid(logits).round()
        expand = preds[:,:,None].repeat(1, 1, self.inputs_per_block, 1, 1)
        expand = expand.reshape()
        mask = expand.type(torch.bool)  # [batch X in_channel * k_h * k_w X h X w]

        # if route_forward, set input patches to zero, otherwise just cut the gradient graph
        out = torch.zeros_like(flat)
        if self.route_forward:
            out[mask] = flat[mask]
        else:
            out[~mask] = flat.data[~mask]
            out[mask] = flat[mask]

        return self.conv(flat)


class FeatureConsolidator(nn.Module):

    def __init__(self):
        super(FeatureConsolidator, self).__init__()
        self.stitching_layer = StitchingLayer()
        self.dropout = AdaptiveDropout()


class ConsolidateArgs(IncrTrainingArgs):
    ARGS = {
        'layer':
            Argument('--layer', type=str, nargs='*', help='specify the layer to consolidate'),
        'num_features':
            Argument('--num-features', type=int, help='specify the number of features at the layer of consolidation'),
        'num_samples':
            Argument('--num-sample', type=int, default=5000,
                     help='specify number of samples for each data subset in consolidation experiments'),
        'single_task':
            Argument('--single-task', action='store_true',
                     help='perform experiment on separate subsets of the same task'),
        'incremental':
            Argument('--incremental', action='store_true',
                     help='incrementally adapt params rather than consolidate after-the-fact'),
        'experiment_id':
            Argument('--experiment-id', type=str, default='diff_task', help='experiment id used for saving models'),
        'reset_bn':
            Argument('--reset-bn', action='store_true',
                     help='disable affine layer and stats tracking for all bn layers'),
        'superimpose':
            Argument('--superimpose', action='store_true', help='experiment with superimposed convolutions'),
        'l2':
            Argument('--l2', action='store_true',
                     help='add L2 penalty to regularize params toward previous params during subsequent task training'),
        'l2_weight':
            Argument('--l2-weight', type=float, default=0.1, help='weight for the L2 penalty'),
        'regularization':
            Argument('--regularization', type=str, default='none', choices=['none', 'l2', 'ewc', 'si', 'mas'],
                     help='choice of l2 regularization scheme for incremental training'),
        'weight_sup_method':
            Argument('--weight-sup-method', type=str, default='none', choices=['none', 'grad', 'avg'],
                     help='how to apply regularization weighting to combine weight averaged loss with model loss'),
        'redundant_groups':
            Argument('--redundant-groups', type=int, default=1, help='number of redundant groups to use'),
        'drop_groups':
            Argument('--drop-groups', action='store_true', help='drop redundant conv blocks')
    }


class DropoutArgs(ConsolidateArgs):
    ARGS = {
        'drop_lr':
            Argument('--drop-lr', type=float, default=0.01, help='step size for fitting dropout weight'),
        'suppress_rate':
            Argument('--suppress-rate', type=float, default=0.02,
                     help='coefficient used to weight negative entropy loss to force multiple features towards zero'),
        'drop_momentum':
            Argument('--drop-momentum', type=float, default=0.9, help='momentum for dropout weight optimization'),
        'drop_nesterov':
            Argument('--drop-nesterov', action='store_true',
                     help='whether to use nesterov optimization on dropout weight'),
        'drop_weight_decay':
            Argument('--drop-weight-decay', type=float, default=5e-4,
                     help='weight decay for dropout weight optimization'),
        'drop_adam':
            Argument('--drop-adam', action='store_true', help='use Adam for dropout weight optimization'),
        'suppress_func':
            Argument('--suppress-func', type=str, default='normal',
                     choices=['normal', 'piecewise-exponential', 'relu', 'exp'],
                     help='specify the function to use for obtaining suppression weighting scheme'),
    }


class ExperimentArgs(DropoutArgs, ConsolidateArgs):
    ARGS = {
        'experiment':
            Argument('--experiment', type=str, default='consolidate', choices=['consolidate', 'dropout']),
        'fit_bn_stats':
            Argument('--fit-bn-stats', action='store_true', help='fit bn moving averages during training experiments')
    }


if __name__ == '__main__':
    main()

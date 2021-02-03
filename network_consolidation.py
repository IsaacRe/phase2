import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from experiment_utils.utils.layer_injection import LayerInjector
from args import *
from experiment_utils.train_models import test, train, get_dataloaders_incr, get_dataloaders, get_subset_data_loaders
from experiment_utils.utils.helpers import find_network_modules_by_name
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


def consolidate_multi_task(data_args, train_args, model, device=0):
    train_loaders, val_loaders, test_loaders = get_dataloaders_incr(data_args, load_test=True)
    _, _, test_ldr = get_dataloaders(data_args, load_train=False)

    reinit_layers = find_network_modules_by_name(model, train_args.layer)

    model.eval()
    # if not updating bn layer during training, disable model's train mode
    if not train_args.fit_bn_stats:
        model.train = lambda *args, **kwargs: None

    # test pretrained model accuracy
    pt_accuracies = []
    for i, test_loader in enumerate(test_loaders):
        c, t = test(model, test_loader, device=device, multihead=True)
        acc = (c.sum() / t.sum()).item()
        print('Pretrained model accuracy for task %d: %.2f' % (i, acc * 100.))
        pt_accuracies += [acc]

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

    base_dir = 'models/consolidation_experiments/diff_task/'
    base_path = base_dir + '%d-layer/' % len(train_args.layer)

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
        pass

    # save pretrained parameterization of the layer
    save_layer(base_path, suffix='-full.pth')

    # reinitialize the layer
    for layer in reinit_layers:
        layer.reset_parameters()
    save_layer(base_path, suffix='-reinit.pth')"""

    accuracies = []
    # train separately on each subtask
    """for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        train(train_args, model, train_loader, val_loader, device=device, optimize_modules=reinit_layers,
              multihead=True)
        model.eval()
        accs = []
        accuracies += [accs]
        for j, test_loader in enumerate(test_loaders):
            c, t = test(model, test_loader, device=device, multihead=True)
            acc = (c.sum() / t.sum()).item()
            accs += [acc]
            print('Task-%d-trained model accuracy for task %d: %.2f' % (i, j, acc * 100.))

        # save trained layer
        save_layer(base_path, suffix='-task_%d.pth' % i)

        # reinitialize the layer
        load_layer(base_path, suffix='-reinit.pth')"""

    # consolidate using kernel averaging
    print('Consolidating separately trained layers...')

    threshold = 0.3
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
        layer.cuda()

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

        # use different reinitialization
        #load_layer(base_path + 'reinit.pth')
        reinit_layer.reset_parameters()
        save_layer(base_path + 'reinit_%d.pth' % (i + 1))

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
                     help='perform experiment on separate subsets of the same task')
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

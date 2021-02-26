from os.path import join
import numpy as np
import torch
from experiment_utils.train_models import get_dataloaders, get_dataloaders_incr, train, test, train_batch_multihead
from experiment_utils.argument_parsing import *
from experiment_utils.utils.helpers import set_torchvision_network_module
from args import *
from model import resnet18, lrm_resnet18, LRMResNetV2
from network_consolidation import apply_module_method_if_exists, collect_l2_weight, SuperConv, L2Conv, ExperimentArgs


def append_to_file(filepath: str, s: str):
    split = filepath.split('.')
    split1 = '.'.join(split[:-1])
    return '.'.join([split1 + s, split[-1]])


def set_task(model, task_id):
    if hasattr(model, 'set_task_id'):
        model.set_task_id(task_id)


def reset_bn(model):
    def build_bn(bn):
        return torch.nn.BatchNorm2d(bn.num_features, eps=bn.eps, momentum=bn.momentum, affine=False,
                                    track_running_stats=False)
    for name, module in model.named_modules():
        if type(module) == torch.nn.BatchNorm2d:
            bn = build_bn(module)
            set_torchvision_network_module(model, name, bn)


def train_incr(args: IncrTrainingArgs, model, train_loaders, val_loaders, device=0):
    # single run-through of all exposures
    acc_save_path = args.acc_save_path
    model_save_path = args.model_save_path
    running_test_results = [[] for _ in range(1, len(train_loaders) + 1)]
    model.active_outputs = []

    # set l2 flag
    if args.regularization != 'none':
        args.l2 = True

    # remove affine layer and stats tracking from all batchnorm layers
    if args.reset_bn:
        reset_bn(model)

    optimize_modules = None
    if args.superimpose:
        model.superimpose = apply_module_method_if_exists(model, 'superimpose')
        model.load_superimposed_weight = apply_module_method_if_exists(model, 'load_superimposed_weight')
        model.update_component = apply_module_method_if_exists(model, 'update_component')
        model.scale_supconv_grads = apply_module_method_if_exists(model, 'scale_grad')

        def build_super_conv(conv):
            return SuperConv(conv.in_channels, conv.out_channels, conv.kernel_size, bias=conv.bias is not None,
                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)

        optimize_modules = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.Conv2d:
                sup_conv = build_super_conv(module).to(module.weight.device)
                set_torchvision_network_module(model, name, sup_conv)
                optimize_modules += [sup_conv]
        optimize_modules += [model.fc]

    elif args.regularization != 'none':
        model.update_previous_params = apply_module_method_if_exists(model, 'update_previous_weight')

        def build_l2_conv(conv):
            return L2Conv(conv.in_channels, conv.out_channels, conv.kernel_size, bias=conv.bias is not None,
                          stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)

        optimize_modules = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.Conv2d:
                sup_conv = build_l2_conv(module).to(module.weight.device)
                set_torchvision_network_module(model, name, sup_conv)
                optimize_modules += [sup_conv]
        optimize_modules += [model.fc]

    for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        if args.exposure_reinit:
            init_state = torch.load(join(args.model_save_dir, append_to_file(model_save_path, 'init')))
            model.cpu().load_state_dict(init_state)
            model.cuda()

        # update active (used) model outputs
        # TODO generalize for exposure repetition
        model.active_outputs += train_loader.classes
        set_task(model, i)

        args.acc_save_path = append_to_file(acc_save_path, '-exp%d' % (i + 1))
        args.model_save_path = append_to_file(model_save_path, '-exp%d' % (i + 1))
        train(args, model, train_loader, *val_loaders[:i+1], device=device, multihead=args.multihead, fc_only=False, #i > 0
              optimize_modules=optimize_modules)

        # update regularization weighting scheme
        if args.regularization not in ['none', 'l2']:
            collect_l2_weight(model, train_loader, method=args.regularization, device=device)

        """
        print('Testing over all %d previously learned tasks...' % (i + 1))
        mean_acc = total_classes = 0
        model.eval()
        if args.superimpose:
            model.superimpose(True)
        for j, test_loader in enumerate(val_loaders[:i+1]):
            set_task(model, j)

            correct, total = test(model, test_loader, device=device, multihead=args.multihead)
            accuracy = correct / total * 100.
            running_test_results[j] += [accuracy]
            mean_acc += accuracy.sum()
            total_classes += len(test_loader.classes)
        mean_acc = mean_acc / total_classes
        print("Mean accuracy over all %d previously learned tasks: %.4f" % (i + 1, mean_acc))
        """

        # update component/previous weight
        if args.superimpose:
            model.update_component()
        elif args.regularization != 'none':
            model.update_previous_params()

        """
        if args.save_acc:
            entropy = class_div = None
            if type(model) == LRMResNetV2:
                entropy = model.get_entropy()
                class_div = model.get_class_routing_divergence()
            np.savez(join(args.acc_save_dir, args.incr_results_path),
                     entropy=entropy,
                     class_div=class_div,
                     **{str(k+1): np.stack(acc) for k, acc in enumerate(running_test_results[:i+1])})
                    """


def load_lrm(state=None, n_blocks=1, block_size_alpha=1.0, load_fc=False, strict_load=True, **kwargs):
    if state is not None:
        n_blocks_new, block_size_alpha_new = n_blocks, block_size_alpha
        # get n_blocks and block_size_alpha
        try:
            n_blocks, block_size_alpha = state['param_n_blocks'], state['param_block_size_alpha']
        except KeyError:  # for compatibility with previously saved states
            n_blocks = state['conv1.key_op.weight'].shape[0]
            block_size_alpha = 1.0
            state['param_n_blocks'] = torch.nn.Parameter(torch.LongTensor([n_blocks]), requires_grad=False)
            state['param_block_size_alpha'] = torch.nn.Parameter(torch.FloatTensor([block_size_alpha]),
                                                                 requires_grad=False)

    lrm_model = lrm_resnet18(n_blocks=n_blocks, block_size_alpha=block_size_alpha, **kwargs)

    if state is not None:
        if not load_fc:
            state['fc.weight'], state['fc.bias'] = lrm_model.fc.weight, lrm_model.fc.bias
        lrm_model.load_state_dict(state, strict=strict_load)
        if n_blocks != n_blocks_new or block_size_alpha != block_size_alpha_new:
            lrm_model.restructure_blocks(n_blocks_new, block_size_alpha_new)

    return lrm_model


def main():
    data_args, train_args, model_args = parse_args(IncrDataArgs, ExperimentArgs, AllModelArgs)
    if train_args.batch and not train_args.multihead:
        train_loader, val_loader, test_loader = get_dataloaders(data_args, load_test=False)
    else:
        train_loader, val_loader, test_loader = get_dataloaders_incr(data_args, load_test=False,
                                                                     multihead_batch=train_args.batch)

    state = None
    # load pretrained feature extractor if specified
    if model_args.load_state_path:
        state = torch.load(model_args.load_state_path)

    if model_args.arch == 'resnet18':
        net = resnet18(num_classes=data_args.num_classes, seed=data_args.seed,
                       disable_bn_stats=model_args.disable_bn_stats)
        if state is not None:
            state['fc.weight'], state['fc.bias'] = net.fc.weight, net.fc.bias
            net.load_state_dict(state)
    elif model_args.arch == 'lrm_resnet18':
        net = load_lrm(state=state, num_classes=data_args.num_classes, seed=data_args.seed,
                       disable_bn_stats=model_args.disable_bn_stats, n_blocks=model_args.n_blocks,
                       block_size_alpha=model_args.block_size_alpha, route_by_task=model_args.route_by_task,
                       fit_keys=train_args.fit_keys)

    # save state initialization if we will be reinitializing the model before each new exposure
    if train_args.exposure_reinit:
        torch.save(net.state_dict(), join(train_args.model_save_dir,
                                          append_to_file(train_args.model_save_path, 'init')))
    net.cuda()

    if train_args.batch:
        if train_args.multihead:
            # trains model on batches of data across tasks while enforcing classification predictions to be within task
            train_batch_multihead(train_args, net, train_loader, val_loader, device=0)
            np.savez(join(train_args.acc_save_dir, train_args.incr_results_path),
                     entropy=net.get_entropy(),
                     class_div=net.get_class_routing_divergence())
        else:
            train(train_args, net, train_loader, val_loader, device=0, multihead=False)
    else:
        train_incr(train_args, net, train_loader, val_loader, device=0)


if __name__ == '__main__':
    main()

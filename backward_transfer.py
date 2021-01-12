from os.path import join
import numpy as np
import torch
from tqdm.auto import tqdm
from experiment_utils.train_models import get_dataloaders, get_dataloaders_incr, test, save_model, train_batch_multihead
from experiment_utils.argument_parsing import *
from args import *
from model import resnet18, lrm_resnet18, LRMResNetV2


def get_params_up_to_layer(model, layer):
    fit_params = set()
    reinit_names = set()
    found_layer = False
    for n, p in model.named_parameters():
        if found_layer and layer not in n:
            break
        fit_params = fit_params.union({p})
        reinit_names = reinit_names.union({n})
        if layer in n:
            found_layer = True

    fit_params = list(fit_params)
    assert len(fit_params) > 0, "fit_params is empty"

    return fit_params, reinit_names


def load_lrm(state, reinit_up_to_layer, n_blocks=1, block_size_alpha=1.0, **kwargs):
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

    train_params, reinit_names = get_params_up_to_layer(lrm_model, reinit_up_to_layer)
    for n in reinit_names:
        del state[n]

    lrm_model.load_state_dict(state, strict=False)
    if n_blocks != n_blocks_new or block_size_alpha != block_size_alpha_new:
        lrm_model.restructure_blocks(n_blocks_new, block_size_alpha_new)

    return lrm_model


def train_bt(args: TrainingArgs, model, train_loader, test_loader, transfer_loader, fit_params, device=0,
             multihead=False):
    active_outputs = np.arange(model.fc.out_features)
    if hasattr(model, 'active_outputs'):
        active_outputs = np.array(model.active_outputs)

    model.train()
    def get_optim(lr):
        return torch.optim.SGD(fit_params,
                               lr=lr,
                               nesterov=args.nesterov,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)

    def get_scheduler(optim):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    lr = args.lr
    optim = get_optim(lr)
    if args.use_schedule:
        scheduler = get_scheduler(optim)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, correct = [], []
    total_transfer, correct_transfer = [], []
    mean_losses = []
    torch.manual_seed(args.seed)  # seed dataloader shuffling

    # if multihead, set all classes other than those of current exposure to inactive (outputs wont contribute)
    if multihead:
        classes = np.array(train_loader.classes)
    # otherwise, make only classes that the model hasnt been exposed to yet inactive (retain previously trained outputs)
    else:
        classes = active_outputs

    inactive_classes = np.where(
        (np.arange(model.fc.out_features)[:, None].repeat(len(classes), 1) == classes[None]).sum(axis=1) == 0
    )

    for e in range(args.epochs):
        # check for lr decay
        if args.use_schedule:
            scheduler.step()
        elif e in args.decay_epochs:
            lr /= args.lr_decay
            optim = get_optim(lr)

        print('Beginning epoch %d/%d' % (e + 1, args.epochs))
        losses = []

        for i, x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)

            out[:, inactive_classes] = float('-inf')

            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses += [loss.item()]

        mean_loss = sum(losses) / len(losses)
        print('Mean loss for epoch %d: %.4f' % (e, mean_loss))
        print('Test accuracy for epoch %d:' % e, end=' ')

        mean_losses += [mean_loss]

        model.eval()
        correct_, total_ = test(model, test_loader, device=device, multihead=multihead)
        correct_transfer_, total_transfer_ = test(model, transfer_loader, device=device, multihead=multihead)
        model.train()
        total += [total_]
        total_transfer += [total_transfer_]
        correct += [correct_]
        correct_transfer += [correct_transfer_]
        if args.save_acc:
            np.savez(join(args.acc_save_dir, args.acc_save_path),
                     train_loss=np.array(mean_losses),
                     val_accuracy=np.stack(correct, axis=0) / np.stack(total, axis=0),
                     transfer_accuracy=np.stack(correct_transfer, axis=0) / np.stack(total_transfer, axis=0))
        save_model(model, join(args.model_save_dir, args.model_save_path), device=device)


class BTTrainingArgs(IncrTrainingArgs):
    ARGS = {
        'train_up_to_layer':
            Argument('--train-up-to-layer', type=str, default='conv1',
                     help='final layer to reinitialize and train incrementally')
    }


def main():
    data_args,  train_args, model_args = parse_args(IncrDataArgs, BTTrainingArgs, AllModelArgs)
    train_loader, val_loader, test_loader = get_dataloaders_incr(data_args, load_test=False)

    # unpack train and val loaders for the 1st task, and val loader for 2nd task to evaluate transfer
    train_loader, *_ = train_loader
    val_loader, transfer_loader, *_ = val_loader

    assert model_args.load_state_path, "Must pass path to model to be used for downstream init"


    state = torch.load(model_args.load_state_path)
    if model_args.arch == 'resnet18':
        net = resnet18(num_classes=data_args.num_classes, seed=data_args.seed,
                       disable_bn_stats=model_args.disable_bn_stats)
        #state['fc.weight'], state['fc.bias'] = net.fc.weight, net.fc.bias
        fit_params, reinit_names = get_params_up_to_layer(net, train_args.train_up_to_layer)
        # do not load layers to be reinitialized and trained
        for n in reinit_names:
            del state[n]
        net.load_state_dict(state, strict=False)
    elif model_args.arch == 'lrm_resnet18':
        #TODO update lrm to allow fitting of task-specific routing
        net, fit_params = load_lrm(state, train_args.train_up_to_layer, num_classes=data_args.num_classes,
                                   seed=data_args.seed,
                                   disable_bn_stats=model_args.disable_bn_stats, n_blocks=model_args.n_blocks,
                                   block_size_alpha=model_args.block_size_alpha, route_by_task=model_args.route_by_task)

    net.cuda()
    train_bt(train_args, net, train_loader, val_loader, transfer_loader, fit_params, device=0,
             multihead=train_args.multihead)

    # check activation distributions of data from inactive class
    from experiment_utils.utils.model_tracking import ModuleTracker, TrackingProtocol
    from experiment_utils.utils.helpers import find_network_modules_by_name
    final_layer_name = train_args.train_up_to_layer
    final_layer, = find_network_modules_by_name(net, [final_layer_name])
    tracker = ModuleTracker(TrackingProtocol('out'), **{final_layer_name: final_layer})

    with tracker.track():
        for _, x1, _ in val_loader:
            net(x1.to(0))
            out1 = tracker.gather_module_var(final_layer_name, 'out')
            break

        tracker.clear_data_buffer_all()
        for _, x2, _ in transfer_loader:
            net(x2.to(0))
            out2 = tracker.gather_module_var(final_layer_name, 'out')
            break

    pass


if __name__ == '__main__':
    main()

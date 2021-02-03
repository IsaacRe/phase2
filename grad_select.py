import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from experiment_utils.utils.helpers import find_network_modules_by_name
from experiment_utils.train_models import *
from experiment_utils.utils.model_tracking import ModuleTracker, TrackingProtocol
from experiment_utils.utils.helpers import get_named_modules_from_network
from experiment_utils.utils.hook_management import HookManager
from experiment_utils.argument_parsing import *
from args import *


def train_grad_select(args: TrainingArgs, model, train_loader, test_loader, device=0, multihead=False,
                      fc_only=False, optimize_modules=None):
    active_outputs = np.arange(model.fc.out_features)
    if hasattr(model, 'active_outputs'):
        active_outputs = np.array(model.active_outputs)

    if fc_only:
        optimize_modules = []
    elif optimize_modules is None:
        optimize_modules = [model]

    params = set()
    for module in optimize_modules:
        params = params.union(set(module.parameters()))
    params = list(params)

    model.train()

    def get_optim(lr):
        if args.adam:
            return torch.optim.Adam(params,
                                    lr=lr,
                                    weight_decay=args.weight_decay)
        return torch.optim.SGD(params,
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

            # zero grads for non-selected weights
            for p in model.parameters():
                p.grad[p.zero_grad_mask] = 0

            optim.step()
            optim.zero_grad()
            losses += [loss.item()]

        mean_loss = sum(losses) / len(losses)
        print('Mean loss for epoch %d: %.4f' % (e, mean_loss))
        print('Test accuracy for epoch %d:' % e, end=' ')

        mean_losses += [mean_loss]

        model.eval()
        correct_, total_ = test(model, test_loader, device=device, multihead=multihead)
        model.train()
        total += [total_]
        correct += [correct_]
        if args.save_acc:
            np.savez(join(args.acc_save_dir, args.acc_save_path),
                     train_loss=np.array(mean_losses),
                     val_accuracy=np.stack(correct, axis=0) / np.stack(total, axis=0))
        save_model(model, join(args.model_save_dir, args.model_save_path), device=device)


def grad_select(args, model, train_loader, test_loader):
    # select weights
    xent = nn.CrossEntropyLoss()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('Selecting weights for training...')
    for n, (i, x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(0), y.to(0)
        out = model(x)
        loss = xent(out, y)
        loss.backward()
        if n > args.batches_for_select:
            break

    for n, p in model.named_parameters():
        p.zero_grad_mask = torch.zeros_like(p).type(torch.bool)
        abs_grad = p.grad.abs()
        if args.method == 'grad':
            perc = 0 if 'fc' in n else np.percentile(abs_grad.flatten().cpu().numpy(),
                                                     (1 - args.select_ratio) * 100.)
            p.zero_grad_mask[abs_grad < perc] = True
        else:
            p.zero_grad_mask[torch.BoolTensor(np.random.rand(*p.shape) > args.select_ratio)] = True

    model.zero_grad()

    train_grad_select(args, model, train_loader, test_loader)


class GradSelectArgs(IncrTrainingArgs):
    ARGS = {
        'select_ratio':
            Argument('--select-ratio', type=float, default=0.03, help='ratio of weights to train'),
        'method':
            Argument('--method', type=str, choices=['random', 'grad'], default='grad',
                     help='method of weight selection for training'),
        'batches_for_select':
            Argument('--batches-for-select', type=int, default=1,
                     help='number of batches of training used to initially select weights for updating')
    }


if __name__ == '__main__':
    data_args, model_init_args, args = \
        parse_args(DataArgs, ModelInitArgs, GradSelectArgs)
    train_loader, test_loader, _ = get_dataloaders(data_args)
    model = initialize_model(model_init_args, device=0)
    grad_select(args, model, train_loader, test_loader)

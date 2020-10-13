from os.path import join
import numpy as np
from experiment_utils.train_models import get_dataloaders_incr, train, test
from experiment_utils.argument_parsing import *
from args import IncrTrainingArgs
from model import resnet18


def append_to_file(filepath: str, s: str):
    split = filepath.split('.')
    split1 = '.'.join(split[:-1])
    return '.'.join([split1 + s, split[-1]])


def train_incr(args: IncrTrainingArgs, model, train_loaders, val_loaders, device=0):
    # single run-through of all exposures
    acc_save_path = args.acc_save_path
    model_save_path = args.model_save_path
    running_test_results = [[] for _ in range(1, len(train_loaders) + 1)]
    model.active_outputs = []

    for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        # update active (used) model outputs
        # TODO generalize for exposure repetition
        model.active_outputs += train_loader.classes

        args.acc_save_path = append_to_file(acc_save_path, '-exp%d' % (i + 1))
        args.model_save_path = append_to_file(model_save_path, '-exp%d' % (i + 1))
        train(args, model, train_loader, val_loader, device=device, multihead=args.multihead)

        print('Testing over all %d previously learned tasks...' % (i + 1))
        mean_acc = total_classes = 0
        model.eval()
        for j, test_loader in enumerate(val_loaders[:i+1]):
            correct, total = test(model, test_loader, device=device, multihead=args.multihead)
            accuracy = correct / total * 100.
            running_test_results[j] += [accuracy]
            mean_acc += accuracy.sum()
            total_classes += len(test_loader.classes)
        mean_acc = mean_acc / total_classes
        print("Mean accuracy over all %d previously learned tasks: %.4f" % (i + 1, mean_acc))

        np.savez(join(args.acc_save_dir, args.incr_results_path),
                 **{str(k+1): np.stack(acc) for k, acc in enumerate(running_test_results[:i+1])})


if __name__ == '__main__':
    data_args, train_args = parse_args(IncrDataArgs, IncrTrainingArgs)
    train_loader, val_loader, test_loader = get_dataloaders_incr(data_args, load_test=False)
    net = resnet18(num_classes=data_args.num_classes, seed=data_args.seed).cuda()
    train_incr(train_args, net, train_loader, val_loader, device=0)

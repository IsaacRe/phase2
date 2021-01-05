import torch
from experiment_utils.train_models import get_dataloaders, save_model, test, train
from experiment_utils.argument_parsing import *
from model import resnet18


class LoadModelArgs(ArgumentClass):
    ARGS = {
        'load_model_path':
            Argument('--load-model-path', type=str, default=None, help='path to model file to load at init')
    }


if __name__ == '__main__':
    model_args, data_args, train_args = parse_args(LoadModelArgs, DataArgs, TrainingArgs)
    train_loader, val_loader, test_loader = get_dataloaders(data_args)
    model = resnet18(num_classes=data_args.num_classes, seed=data_args.seed)
    if model_args.load_model_path:
        model.load_state_dict(torch.load(model_args.load_model_path))
    model.cuda()
    train(train_args, model, train_loader, val_loader, device=0)

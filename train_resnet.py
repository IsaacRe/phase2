from experiment_utils.train_models import get_dataloaders, save_model, test, train
from experiment_utils.argument_parsing import *
from model import resnet18


if __name__ == '__main__':
    data_args, train_args = parse_args(DataArgs, TrainingArgs)
    train_loader, val_loader, test_loader = get_dataloaders(data_args)
    model = resnet18(num_classes=data_args.num_classes, seed=data_args.seed)
    model.cuda()
    train(train_args, model, train_loader, val_loader, device=0)

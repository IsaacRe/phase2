from experiment_utils.train_models import get_dataloaders, train
from experiment_utils.argument_parsing import *
from args import LRMV1Args
from model import lrm_resnet18


if __name__ == '__main__':
    data_args, train_args, model_args = parse_args(DataArgs, TrainingArgs, LRMV1Args)
    train_loader, val_loader, test_loader = get_dataloaders(data_args)
    model = lrm_resnet18(**model_args)
    model.cuda()
    train(train_args, model, train_loader, val_loader, device=0)


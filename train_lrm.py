from experiment_utils.train_models import get_dataloaders, save_model, test, train
from experiment_utils.argument_parsing import *
from model import lrm_resnet18


class LRMV1Args(NumClass, Seed):
    ARGS = {
        'n_blocks':
            Argument('--n-blocks', type=int, default=1, help='number of low-rank blocks in each mixture'),
        'block_size_alpha':
            Argument('--block-size-alpha', type=float, default=1.0,
                     help='scales the computed total-memory-preserving block size for each layer. '
                          'Each layer will have block size = nm / (n + m) / n_blocks * block_size_alpha ,'
                          'where n = output dim, m = input dim')
    }


if __name__ == '__main__':
    data_args, train_args, model_args = parse_args(DataArgs, TrainingArgs, LRMV1Args)
    train_loader, val_loader, test_loader = get_dataloaders(data_args)
    model = lrm_resnet18(**model_args)
    model.cuda()
    train(train_args, model, train_loader, val_loader, device=0)


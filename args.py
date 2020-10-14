from experiment_utils.argument_parsing import *


####    Model Args    ##########################################


class ResNetArgs(NumClass, Seed):
    ARGS = {
        'disable_bn_stats':
            Argument('--disable-bn-stats', action='store_true',
                     help='disable the tracking of running statistics by batchnorm layers')
    }


class LRMV1Args(ResNetArgs):
    ARGS = {
        'n_blocks':
            Argument('--n-blocks', type=int, default=1, help='number of low-rank blocks in each mixture'),
        'block_size_alpha':
            Argument('--block-size-alpha', type=float, default=1.0,
                     help='scales the computed total-memory-preserving block size for each layer. '
                          'Each layer will have block size = nm / (n + m) / n_blocks * block_size_alpha ,'
                          'where n = output dim, m = input dim'),
        'route_by_task':
            Argument('--route-by-task', action='store_true',
                     help='allocate separate blocks for each task and route accordingly')
    }


class AllModelArgs(LRMV1Args):
    ARGS = {
        'arch':
            Argument('--arch', type=str, default='resnet18', choices=['resnet18', 'lrm_resnet18'],
                     help='specify model to use'),
        'load_state_path':
            Argument('--load-state-path', type=str, default='',
                     help='specify path to model state file to load pretrained weights')
    }


####    Incremental Training Args    ###################

class IncrTrainingArgs(TrainingArgs):
    ARGS = {
        'num_repetitions':
            Argument('--num-repetitions', type=int, default=1, help='number of passes to take over all exposures'),
        'exposure_reinit':
            Argument('--exposure-reinit', action='store_true', help='reinitialize model before each exposure'),
        'multihead':
            Argument('--multihead', action='store_true', help='use separate classification heads for each exposure'),
        'incr_results_path':
            Argument('--incr-results-path', type=str, default='incr-accuracy.npz',
                     help='save path for accuracy over all incremental exposures as learning progresses')
    }

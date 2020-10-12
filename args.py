from experiment_utils.argument_parsing import *


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

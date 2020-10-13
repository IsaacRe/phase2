from experiment_utils.argument_parsing import *


####    Model Args    ##########################################

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


####    Incremental Training Args    ###################

class IncrTrainingArgs(TrainingArgs):
    ARGS = {
        'num_repetitions':
            Argument('--num-repetitions', type=int, default=1, help='number of passes to take over all exposures'),
        'multihead':
            Argument('--multihead', action='store_true', help='use separate classification heads for each exposure'),
        'incr_results_path':
            Argument('--incr-results-path', type=str, default='incr-accuracy.npz',
                     help='save path for accuracy over all incremental exposures as learning progresses')
    }



import argparse
import collections
import json
import os
import random
import sys
import time
import copy
import uuid
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import yaml
import datasets
import hparams_registry
import algorithms
import numpy.random as random
from lib import misc

from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import ConcatDataset


def running_time(totalsec):
    day = totalsec // (24 * 3600)
    restsec = totalsec % (24 * 3600)
    hour = restsec // 3600
    restsec %= 3600
    minutes = restsec // 60
    restsec %= 60
    seconds = restsec
    print("Total running time: %d days, %d hours, %d minutes, %d seconds." % (day, hour, minutes, seconds))

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

if __name__ == "__main__":
    import os


    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument("--save", default=r".\results", type=str,
                        help="save location")
    parser.add_argument("--gpu", default=r"1", type=str,
                        help="save location")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST_ONLINE")

    parser.add_argument('--algorithm', type=str, default="FairDolce")
    parser.add_argument('--gen_dir', type=str, default="./Model/RCMNIST_model_1.pkl", help="if not empty, the generator of DEDF will be loaded")
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--task', type=str, default="domain_generalization",
        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--image_display_iter', type=int, default=500,
        help='Epochs interval for showing the generated images')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=3,  #3
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[])
    parser.add_argument('--output_dir', type=str, default="train_outputs")
    parser.add_argument('--save_name', type=str, default="RCMNIST_stage1_model_1.pkl")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))




    if args.hparams_seed == 0:

        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, stage=args.stage)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed), stage=args.stage)
    if args.hparams:

        hparams.update(json.loads(args.hparams))




    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"



    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    dataset.input_shape = (3, 28, 28)



    algorithm_class = algorithms.get_algorithm_class(args.algorithm)

    dataset.input_shape=(3,28,28)
    dataset.num_classes =2
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) , hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if args.algorithm == 'FairDolce' and os.path.exists(args.gen_dir) and hparams['stage'] == 1:
        pretext_model = torch.load(args.gen_dir)['model_dict']
        alg_dict = algorithm.state_dict()
        ignored_keys = []
        state_dict = {k: v for k, v in pretext_model.items() if k in alg_dict.keys() and ('id_featurizer' in k or 'gen' in k)}
        alg_dict.update(state_dict)
        algorithm.load_state_dict(alg_dict)
        algorithm_copy = copy.deepcopy(algorithm)
        algorithm_copy.eval()
    else:
        algorithm_copy = None


    n_steps = args.steps or dataset.N_STEPS
    if 'FairDolce' in args.algorithm:
        n_steps = hparams['steps']
    last_results_keys = None
    def save_checkpoint(filename):
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    checkpoint_vals = collections.defaultdict(lambda: [])
    T = None


    for task_i, env_i in enumerate(dataset):


        if T == None:
            T = env_i
        else:
            T = ConcatDataset([T, env_i])
        eval_loader = FastDataLoader(
            dataset=env_i,
            batch_size=len(env_i),
            num_workers=dataset.N_WORKERS)
        eval_loader_name = 'task{}'.format(task_i)



        acc = misc.accuracy(algorithm, eval_loader, None, device, args=args, step=None, flag=True)
        dp = misc.dp(algorithm, eval_loader, None, device, args=args, step=None, flag=True)
        eo = misc.eo(algorithm, eval_loader, None, device, args=args, step=None, flag=True)








        loaders = [InfiniteDataLoader(
            dataset=T,
            weights=None,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            ]
        minibatches_iterator = zip(*loaders)

        multidomain = False
        if task_i >2:
            multidomain = True

        for step in range(start_step, n_steps):
            step_start_time = time.time()
            minibatches_device = [(x.to(device), y.to(device), pos.to(device), z.to(device), z_pos.to(device)) for x, y, pos, z,z_pos in
                                  next(minibatches_iterator)]
            minibatches_device_neg = [(x.to(device), y.to(device), pos.to(device), z.to(device), z_pos.to(device)) for x, y, pos, z, z_pos in
                                      next(minibatches_iterator)]

            step_vals = algorithm.update(minibatches_device, minibatches_device_neg, pretrain_model=algorithm_copy,multidomain = multidomain)
























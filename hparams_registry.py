

import numpy as np
from lib import misc

def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed, stage):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}
    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', True, lambda r: False)
    _hparam('resnet_dropout', 0.0, lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False, True])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if 'MNIST' in dataset:
        _hparam('is_mnist', True, lambda r: True)

    else:
        _hparam('is_mnist', False, lambda r: False)

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 5, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    if 'FairDolce' in algorithm:

        _hparam('is_FairDolce', True, lambda r: True)
        if algorithm == 'DDG_AugMix':
            _hparam('is_augmix', True, lambda r: True)
        else:
            _hparam('is_augmix', False, lambda r: False)
        if 'MNIST' in dataset:
            _hparam('stage', 1, lambda r: stage) # no use

            _hparam('steps', 50, lambda r: 10000)
            _hparam('margin2', 0.025, lambda r: 0.025)
            _hparam('margin1', 0.025, lambda r: 0.025)
            _hparam('recon_id_w', 0.5, lambda r: r.choice([0.1, 0.2, 0.5, 1.0]))
            _hparam('recon_x_w', 0.5, lambda r: r.choice([1., 2., 5., 10.]))
            _hparam('fair_w', 0.5, lambda r: 0.025)

        _hparam('recon_xp_w', 0.5, lambda r: r.choice([1., 2., 5., 10.]))

        _hparam('eta', 0.01, lambda r: 0.05)

    else:

        _hparam('is_FairDolce', False, lambda r: False)



    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif 'FairDolce' in algorithm:
        _hparam('lr', 2e-5, lambda r: 2e-5)
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 1024, lambda r: int(2**r.uniform(3, 9)) )
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif 'FairDolce' in algorithm:
        _hparam('batch_size', 1024, lambda r: 4)   # this one
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)) )
    else:
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)) )

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5) )
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5) )
    elif 'FairDolce' in algorithm:
        _hparam('lr_g', 1e-4, lambda r: 10**r.uniform(-5, -3.5) )

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5) )
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5) )
    elif  'FairDolce' in algorithm:
        _hparam('lr_d', 1e-4, lambda r: 10**r.uniform(-5, -3.5) )

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN','FairDolce']:
        _hparam('weight_decay_g', 0.0005, lambda r: 10**r.uniform(-6, -2) )

    return hparams

def default_hparams(algorithm, dataset, stage=0):
    return {a: b for a,(b,c) in
        _hparams(algorithm, dataset, 0, stage).items()}

def random_hparams(algorithm, dataset, seed, stage=0):
    return {a: c for a,(b,c) in _hparams(algorithm, dataset, seed, stage).items()}

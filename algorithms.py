

import torch
import torch.nn as nn
import torch.nn.functional as F


import networks
from lib.misc import random_pairs_of_minibatches, Augmix,ddp


ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'DDG_AugMix',
    'FairDolce'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.device = device

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):

        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, device)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )


    def update(self, minibatches, unlabeled=None):

        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)




class FairDolce(ERM):
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, device='cpu'):

        super(FairDolce, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, device)
        self.hparams = hparams
        self.class_balance=hparams['class_balanced']
        self.iteration = 0
        self.id_featurizer = self.featurizer
        self.dis_id = self.classifier
        self.gen = networks.VAEGen(input_dim = input_shape[0])

        self.dis_img = networks.MsImageDis(hparams=hparams) 
        self.recon_xp_w = hparams['recon_xp_w']
        self.fair_w =hparams['fair_w']
        self.recon_id_w  = hparams['recon_id_w']
        self.margin1 = hparams['margin1']
        self.margin2 = hparams['margin2']
        self.eta = hparams['eta']
        

        self.optimizer_gen = torch.optim.Adam([p for p in list(self.gen.parameters())  if p.requires_grad], lr=self.hparams['lr_g'], betas=(0, 0.999), weight_decay=self.hparams['weight_decay_g'])

        self.id_criterion = nn.CrossEntropyLoss()
        self.fair_criterion = ddp
        self.dom_criterion = nn.CrossEntropyLoss()
        

    def recon_criterion(self, input, target, reduction=True):
            diff = input - target.detach()
            B,C,H,W = input.shape
            if reduction == False:
                return torch.mean(torch.abs(diff[:]).view(B,-1),dim=-1)
            return torch.mean(torch.abs(diff[:]))
    
    def train_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()


    def forward(self, x_a, x_b, xp_a, xp_b):
        '''
            inpus:
                x_a, x_b: image from dataloader a,b
                xp_a, xp_b: positive pair of x_a, x_b
        '''

        s_a = self.gen.encode(x_a)
        s_b = self.gen.encode(x_b)

        f_a, x_fa = self.id_featurizer(x_a, self.hparams['stage'])

        p_a = self.dis_id(x_fa)
        f_b, x_fb = self.id_featurizer(x_b, self.hparams['stage'])
        p_b = self.dis_id(x_fb)
        fp_a, xp_fa = self.id_featurizer(xp_a, self.hparams['stage'])
        pp_a = self.dis_id(xp_fa)
        fp_b, xp_fb = self.id_featurizer(xp_b, self.hparams['stage'])
        pp_b = self.dis_id(xp_fb)

        x_ba = self.gen.decode(s_b, f_a)
        x_ab = self.gen.decode(s_a, f_b)
        x_a_recon = self.gen.decode(s_a, f_a)
        x_b_recon = self.gen.decode(s_b, f_b) 
        x_a_recon_p = self.gen.decode(s_a, fp_a)
        x_b_recon_p = self.gen.decode(s_b, fp_b)

        return x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p    



    def gen_update(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b,  l_a, l_b, hparams):

        self.optimizer_gen.zero_grad()
        self.optimizer.zero_grad()





        self.recon_a2a, self.recon_b2b = self.recon_criterion(x_a_recon_p, x_a, reduction=False), self.recon_criterion(x_b_recon_p, x_b, reduction=False)
        self.loss_gen_recon_p =  torch.mean(torch.max(self.recon_a2a-self.margin2, torch.zeros_like(self.recon_a2a)))+ torch.mean(torch.max(self.recon_b2b-self.margin2, torch.zeros_like(self.recon_b2b))) # L recon


        if not hparams['is_mnist']:
            _, x_fa_recon = self.id_featurizer(x_ab); p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ba); p_b_recon = self.dis_id(x_fb_recon)
        else:
            _, x_fa_recon = self.id_featurizer(x_ba); p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ab); p_b_recon = self.dis_id(x_fb_recon)            
        self.loss_id = self.id_criterion(p_a, l_a) + self.id_criterion(p_b, l_b) +  self.id_criterion(pp_a, l_a) + self.id_criterion(pp_b, l_b)
        self.fair_id = torch.mean(self.fair_criterion(self.z_a,p_a)+ self.fair_criterion(self.z_b,p_b)+self.fair_criterion(self.z_a_pair,pp_a)+ self.fair_criterion(self.z_b_pair,pp_b))
        self.loss_gen_recon_id = self.id_criterion(p_a_recon, l_a) + self.id_criterion(p_b_recon, l_b) # L inv
        recon_p =  self.loss_gen_recon_p   #torch.mean(self.recon_a2a)
        inv_p = self.loss_gen_recon_id
        fair_p = self.fair_id
        self.step(recon_p,inv_p,fair_p)


        self.loss_gen_total = self.loss_id +\
                self.recon_xp_w * self.loss_gen_recon_p +\
                self.recon_id_w * self.loss_gen_recon_id  +\
                self.fair_w * self.fair_id

        self.loss_gen_total.backward()
        self.optimizer_gen.step()
        self.optimizer.step()

    def gen_update_easy(self, x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, x_a, x_b,  l_a, l_b, hparams):


        self.optimizer_gen.zero_grad()
        self.optimizer.zero_grad()



        self.loss_gen_recon_p = 0

        if not hparams['is_mnist']:
            _, x_fa_recon = self.id_featurizer(x_ab); p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ba); p_b_recon = self.dis_id(x_fb_recon)
        else:
            _, x_fa_recon = self.id_featurizer(x_ba); p_a_recon = self.dis_id(x_fa_recon)
            _, x_fb_recon = self.id_featurizer(x_ab); p_b_recon = self.dis_id(x_fb_recon)
        self.loss_id = self.id_criterion(p_a, l_a) + self.id_criterion(p_b, l_b)
        self.fair_id = torch.mean(self.fair_criterion(self.z_a,p_a)+ self.fair_criterion(self.z_b,p_b))


        self.loss_gen_recon_id = self.id_criterion(p_a_recon, l_a) + self.id_criterion(p_b_recon, l_b)
        recon_p =  self.loss_gen_recon_p
        inv_p = self.loss_gen_recon_id
        fair_p = self.fair_id
        self.step_easy(inv_p,fair_p)


        self.loss_gen_total = self.loss_id +\
                self.recon_xp_w * self.loss_gen_recon_p +\
                self.recon_id_w * self.loss_gen_recon_id  +\
                self.fair_w * self.fair_id

        self.loss_gen_total.backward()
        self.optimizer_gen.step() # auto encoder(encoder decoder)
        self.optimizer.step() # cls


    def update(self, minibatches, minibatches_neg, pretrain_model=None, unlabeled=None, iteration=0,multidomain = False):

        images_a = torch.cat([x for x, y, pos, z, z_p in minibatches])
        labels_a = torch.cat([y for x, y, pos, z, z_p in minibatches])
        pos_a = torch.cat([pos for x, y, pos, z, z_p in minibatches])
        self.z_a = torch.cat([z for x, y, pos, z, z_p in minibatches])
        self.z_a_pair= torch.cat([z_p for x, y, pos, z, z_p in minibatches])

        images_b = torch.cat([x for x, y, pos, z, z_p in minibatches_neg])
        labels_b = torch.cat([y for x, y, pos, z, z_p in minibatches_neg])
        pos_b = torch.cat([pos for x, y, pos, z, z_p in minibatches_neg])
        self.z_b = torch.cat([z for x, y, pos, z, z_p in minibatches_neg])
        self.z_b_pair = torch.cat([z_p for x, y, pos, z, z_p in minibatches])


        x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = self.forward(images_a, images_b, pos_a, pos_b)


        if multidomain:
            self.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, labels_a, labels_b, self.hparams)
        else:
            self.gen_update_easy(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p,
                            x_b_recon_p, images_a, images_b, labels_a, labels_b, self.hparams)
        return {
                    'loss_cls': self.loss_id.item(),
                    "fair_ddp": float(self.fair_id),
                    'loss_recon_p': self.loss_gen_recon_p.item() if type(self.loss_gen_recon_p)!=int else self.loss_gen_recon_p,
                    'loss_gen_recon_id': self.loss_gen_recon_id.item(),
                    'loss_total': float(self.loss_gen_total),
                    'fair_w':float(self.fair_w),
                    'recon_xp_w': float(self.recon_xp_w),
                    'recon_id_w': float(self.recon_id_w)
        }
                    


    def predict(self, x):
        return self.dis_id(self.id_featurizer(x)[-1])

    def step(self, recon_p=None,inv_p=None,fair_p=None):

        self.iteration += 1

        if recon_p is not None:
            self.recon_xp_w = min(max(self.recon_xp_w + self.eta * (recon_p.item() - self.margin2), 0), 1)
        if inv_p is not None:
            self.recon_id_w = min(max(self.recon_id_w + self.eta * (inv_p.item()), 0), 1)
        if fair_p is not None:
            self.fair_w = min(max(self.fair_w + self.eta * (fair_p.item() - self.margin1), 0), 3)


    def step_easy(self, inv_p=None, fair_p=None):  # recon_p  t2 inv_p : t3 fair_p t1

        self.iteration += 1

        if inv_p is not None:
            self.recon_id_w = min(max(self.recon_id_w + self.eta * (inv_p.item()), 0), 1)
        if fair_p is not None:
            self.fair_w = min(max(self.fair_w + self.eta * (fair_p.item() - self.margin1), 0), 3)



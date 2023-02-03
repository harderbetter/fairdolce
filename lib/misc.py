

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile


import numpy as np
import torch
import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):

    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def sample_tuple_of_minibatches(minibatches, device):
    disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])
    perm = torch.randperm(len(minibatches)).tolist()
    tuples = []
    labels = np.array([minibatches[i][1] for i in range(len(minibatches))])
    
    for i in range(len(minibatches)):

        x, y, d = minibatches[i][0], minibatches[i][1], disc_labels[i]
        x_n, y_n, d_n = minibatches[perm[i]][0], minibatches[perm[i]][1], disc_labels[perm[i]]
        while y_n == y:
            i = perm[i]
            x_n, y_n = minibatches[perm[i]][0], minibatches[perm[i]][1], disc_labels[perm[i]]
        
        pos_ind = np.argwhere(labels == y); pos_n_ind = np.where(labels == y_n)
        x_p, x_np = minibatches[pos_ind[0]][0], minibatches[pos_n_ind[0]][0]

        tuples.append((x, y, d, x_p), (x_n, y_n, d_n, x_np))

    return tuples

def plot_confusion(matrix):
    pass

def accuracy(network, loader, weights, device, args=None, step=None, flag=False):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        if flag:

            for x, y, _, _,_ in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)
                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset : weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)

                if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
        else:
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                p = network.predict(x)


                if weights is None:
                    batch_weights = torch.ones(len(x))
                else:
                    batch_weights = weights[weights_offset : weights_offset + len(x)]
                    weights_offset += len(x)
                batch_weights = batch_weights.to(device)

                if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
                else:
                    correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
                total += batch_weights.sum().item()
                
    network.train()

    return correct / total

def dp(network, loader, weights, device, args=None, step=None, flag=False):
    count1 = 0
    count2 = 0


    network.eval()
    with torch.no_grad():
        for x, y, _, z,_ in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            p = network.predict(x)
            # print(z)
            # print(y)
            for i in range(y.size(0)):
                if z[i].item() == -1.0:
                    count1 += 1
                    if p[i].argmax(0).item()  == 1:
                        count2 += 1
    network.train()

    try:
        dp = abs(1 - count2 * 1.0 / count1)  # 1
    except:
        dp = 0
    return dp


def ddp(z,yhat):

    length = len(z)
    countz =0;

    for item in z:
        if item==1:

            countz +=1;
    p1 = (countz*1.0)/length;
    if length ==0 or p1 ==0:

        return torch.tensor(0.0001)

    sum = 0;
    for i in range(len(z)):
        cur_z = z[i]
        cur_y_hat = yhat[i]

        sum += ((cur_z + 1) / 2 - p1) * cur_y_hat
    sum = sum * 1/(p1*(1 - p1));
    output = sum / length
    output = torch.abs(output)


    return output

def eo(network, loader, weights, device, args=None, step=None, flag=False):
    count1 = 0
    count2 = 0

    weights_offset = 0

    network.eval()
    with torch.no_grad():
        # for x, y in loader:
        for x, y, _, z,_ in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            p = network.predict(x)

            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)

            for i in range(y.size(0)):

                if z[i].item() == -1.0 and y[i].item() == 1:

                    count1 += 1
                    if p[i].argmax(0).item() == 1:
                        count2 += 1

    network.train()

    try:
        eop = abs(1 - count2 * 1.0 / count1)
    except:
        eop = 0
    return eop




class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()



def Augmix(x, preprocess, no_jsd):
    pass

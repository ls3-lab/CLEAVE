import sys
import numpy
import argparse
import os.path as osp
import random
import nni
import datetime
import os
from types import SimpleNamespace

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch_geometric.utils import dropout_adj

from simple_param.sp import SimpleParam
from pGRACE.model import GRACE_Encoder, GRACE_Model
from pGRACE.functional import drop_feature

from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation
from src.data.loader import get_dataset
from src.data.splits import get_split
from src.evaluations.retrieval_validation import compute_avg_rank_change

import yaml
from yaml import SafeLoader
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter as t

#____________________________________________________________________________________________________________________________

DATETIME_NOW = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def train(model: GRACE_Model, x, edge_index, optimizer, param):

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # edge dropout (dropout_adj returns (edge_index, edge_attr))
    edge_index_1, _ = dropout_adj(edge_index, p=param['drop_edge_rate_1'], force_undirected=False)
    edge_index_2, _ = dropout_adj(edge_index, p=param['drop_edge_rate_2'], force_undirected=False)

    # feature dropout
    x_1 = drop_feature(x, param['drop_feature_rate_1'])
    x_2 = drop_feature(x, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu())



def test(model: GRACE_Model, data, dataset, split, num_epochs=3000):
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    evaluator = MulticlassEvaluator()
    return log_regression(
        z,
        dataset,
        evaluator,
        num_epochs=num_epochs,
        dataset_name=args.dataset,
        preload_split=split,
    )['acc']


# python train.py --dataset Cora

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default=None)
    parser.add_argument('--seed', type=int, default=38108)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--output', type=str, default='../../../cache/checkpoints/grace/')

    default_param = {
        "learning_rate": 0.001,
        "num_hidden": 256,
        "num_proj_hidden": 256,
        "activation": "prelu",
        "base_model": "GCNConv",
        "num_layers": 2,
        "drop_edge_rate_1": 0.2,
        "drop_edge_rate_2": 0.0,
        "drop_feature_rate_1": 0.3,
        "drop_feature_rate_2": 0.2,
        "tau": 0.9,
        "num_epochs": 200,
        "weight_decay": 0.00001,
    }



    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
    if args.param is None:
        args.param = f"local:{args.dataset.lower()}.json"
    base_path = os.path.join(args.output, args.dataset)
    os.makedirs(base_path, exist_ok=True)
    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    print(param)
    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    device = torch.device(args.device)


    
    # Use the project loader (src/data/loader.py) so data lives in
    # <project>/cache/datasets/. Loader uses paths relative to project root,
    # so chdir there for the call.
    _old_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        dataset = get_dataset(SimpleNamespace(name=args.dataset))
    finally:
        os.chdir(_old_cwd)

    data = dataset[0]
    data = data.to(device)

    # generate split
    split = get_split(data, dataset_name=args.dataset, mode="train")
    train_mask, val_mask, test_mask = split

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)
        train_mask, val_mask, test_mask = split


    encoder = GRACE_Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE_Model(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])


    start = t()
    prev = start
    log = args.verbose.split(',')

    best_ac = 0.0
    best_avg_rank_change = float('inf')

    for epoch in range(1, param['num_epochs'] + 1):
        loss = train(model, data.x, data.edge_index, optimizer, param)

        now = t()
        if 'train' in log and epoch % 500 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

        if epoch % 100 == 0:
            # Linear-probe classification evaluation on frozen embeddings.
            ac = test(model, data, dataset, split=split)
            if ac > best_ac:
                best_ac = ac
                #torch.save(model.state_dict(), os.path.join(base_path, 'v2_pgrace_best_ac.pth'))    #pgrace_best_ac
                #torch.save(encoder.state_dict(), os.path.join(base_path, 'v2_encoder_best_ac.pth')) #encoder_best_ac
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc={ac:.4f} (best={best_ac:.4f})')

            # Retrieval validation: average node rank change on val set
            avg_rank_change = compute_avg_rank_change(
                model, data, val_mask, device
            )
            if avg_rank_change < best_avg_rank_change:
                best_avg_rank_change = avg_rank_change
                torch.save(model.state_dict(), os.path.join(base_path, 'pgrace_gcn.pth'))    #pgrace_best_retrieval
                torch.save(encoder.state_dict(), os.path.join(base_path, 'encoder.pth')) #encoder_best_retrieval
            print(f'(R) | Epoch={epoch:04d}, avg_rank_change={avg_rank_change:.4f} (best={best_avg_rank_change:.4f})')

    print("=== Final ===")
    acc = test(model, data, dataset, split=split) 
 

    #torch.save(model.state_dict(), os.path.join(base_path, 'v2_pgrace.pth'))
    #torch.save(encoder.state_dict(), os.path.join(base_path, 'v2_encoder.pth'))
    print('Saving the models in\n{}\n{}'.format(os.path.join(base_path, 'pgrace_gcn_attack.pth'), os.path.join(base_path, 'v2_encoder.pth')))
    print(f'Best classification model saved at {os.path.join(base_path, "encoder_attack.pth")}')
    print(f'Best ac: {best_ac:.4f}')
    print(f'Best retrieval model saved at {os.path.join(base_path, "pgrace_gcn_attack.pth")}')
    print(f'Best avg_rank_change: {best_avg_rank_change:.4f}')
    if 'final' in log:
        print(f'{acc}')







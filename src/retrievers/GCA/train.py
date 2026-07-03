import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy
import argparse
import os.path as osp
import random
import nni
import datetime
import os

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    compute_pr, eigenvector_centrality
from src.data.splits import get_split
from pGRACE.dataset import get_dataset
from src.evaluations.retrieval_validation import compute_avg_rank_change


DATETIME_NOW = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def train():
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index)
    print(z.shape)
    evaluator = MulticlassEvaluator()
    acc = log_regression(z, dataset, evaluator, num_epochs=3000,
                         dataset_name=args.dataset, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc

# python train.py --dataset Cora
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--output', type=str, default='../../../cache/checkpoints/gca/')

    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }



    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()
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

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    device = torch.device(args.device)

    path = osp.expanduser('../../../datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

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

    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    log = args.verbose.split(',')

    best_avg_rank_change = float('inf')

    for epoch in range(1, param['num_epochs'] + 1):
        loss = train()
        if 'train' in log and epoch % 500 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % 100 == 0:
            acc = test()

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

            # Retrieval validation: average node rank change on val set
            avg_rank_change = compute_avg_rank_change(
                model, data, val_mask, device
            )
            if avg_rank_change < best_avg_rank_change:
                best_avg_rank_change = avg_rank_change
                torch.save(model.state_dict(), os.path.join(base_path, f'pgrace_gcn_best_retrieval.pth'))
            print(f'(R) | Epoch={epoch:04d}, avg_rank_change={avg_rank_change:.4f} (best={best_avg_rank_change:.4f})')

    acc = test(final=True)
    torch.save(model.state_dict(), os.path.join(base_path, f'pgrace_gcn_{DATETIME_NOW}.pth'))
    print('Saving the model in {}'.format(os.path.join(base_path, f'pgrace_gcn_{DATETIME_NOW}.pth')))
    print(f'Best retrieval model saved at {os.path.join(base_path, "pgrace_gcn_best_retrieval.pth")}')
    print(f'Best avg_rank_change: {best_avg_rank_change:.4f}')
    if 'final' in log:
        print(f'{acc}')




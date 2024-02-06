"""
Accurate Node Feature Estimation with Structured Variational Graph Autoencoder
(KDD 2022)

Authors:
- Jaemin Yoo (jaeminyoo@cmu.edu), Carnegie Mellon University
- Hyunsik Jeon (jeon185@snu.ac.kr), Seoul National University
- Jinhong Jung (jinhongjung@jbnu.ac.kr), Jeonbuk National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

import argparse
import io
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd

from data import load_data, is_continuous, is_large, to_edge_tensor
from models.svga import SVGA, StochasticSVGA
from utils import str2bool, to_device

import gc
gc.collect()
torch.cuda.empty_cache()

def sample_y_nodes(num_nodes, y, y_ratio, seed):
    """
    Sample nodes with observed labels.
    """
    if y_ratio == 0:
        y_nodes = None
        y_labels = None
    elif y_ratio == 1:
        y_nodes = torch.arange(num_nodes)
        y_labels = y[y_nodes]
    else:
        y_nodes, _ = train_test_split(np.arange(num_nodes), train_size=y_ratio, random_state=seed,
                                      stratify=y.numpy())
        y_nodes = torch.from_numpy(y_nodes)
        y_labels = y[y_nodes]
    return y_nodes, y_labels


@torch.no_grad()
def to_f1_score(input, target, epsilon=1e-8):
    """
    Compute the F1 score from a prediction.
    """
    assert (target < 0).int().sum() == 0
    tp = ((input > 0) & (target > 0)).sum()
    fp = ((input > 0) & (target == 0)).sum()
    fn = ((input <= 0) & (target > 0)).sum()
    return (tp / (tp + (fp + fn) / 2 + epsilon)).item()


@torch.no_grad()
def to_recall(input, target, k=10):
    """
    Compute the recall score from a prediction.
    """
    pred = input.topk(k, dim=1, sorted=False)[1]
    row_index = torch.arange(target.size(0))
    target_list = []
    for i in range(k):
        target_list.append(target[row_index, pred[:, i]])
    num_pred = torch.stack(target_list, dim=1).sum(dim=1)
    num_true = target.sum(dim=1)
    return (num_pred[num_true > 0] / num_true[num_true > 0]).mean().item()


@torch.no_grad()
def to_ndcg(input, target, k=10, version='sat'):
    """
    Compute the NDCG score from a prediction.
    """
    if version == 'base':
        return ndcg_score(target, input, k=k)
    elif version == 'sat':
        device = target.device
        target_sorted = torch.sort(target, dim=1, descending=True)[0]
        pred_index = torch.topk(input, k, sorted=True)[1]
        row_index = torch.arange(target.size(0))
        dcg = torch.zeros(target.size(0), device=device)
        for i in range(k):
            dcg += target[row_index, pred_index[:, i]] / np.log2(i + 2)
        idcg_divider = torch.log2(torch.arange(target.size(1), dtype=float, device=device) + 2)
        idcg = (target_sorted / idcg_divider).sum(dim=1)
        return (dcg[idcg > 0] / idcg[idcg > 0]).mean().item()
    else:
        raise ValueError(version)


@torch.no_grad()
def to_r2(input, target):
    """
    Compute the CORR (or the R square) score from a prediction.
    """
    a = ((input - target) ** 2).sum()
    b = ((target - target.mean(dim=0)) ** 2).sum()
    return (1 - a / b).item()


@torch.no_grad()
def to_rmse(input, target):
    """
    Compute the RMSE score from a prediction.
    """
    return ((input - target) ** 2).mean(dim=1).sqrt().mean().item()


def print_log(epoch, loss_list, acc_list):
    """
    Print a log during the training.
    """
    print(f'{epoch:5d}', end=' ')
    print(' '.join(f'{e:.4f}' for e in loss_list), end=' ')
    print(' '.join(f'{e:.4f}' for e in acc_list))


@torch.no_grad()
def evaluate_last(data, model, edge_index, test_nodes, true_features):
    """
    Evaluate a prediction model after the training is done.
    """
    model.eval()
    x_hat, _ = model(edge_index)
    x_hat = x_hat[test_nodes]
    x_true = true_features[test_nodes]

    if is_continuous(data):
        return [to_r2(x_hat, x_true), to_rmse(x_hat, x_true)]
    elif data == "pamap2":
        return [to_r2(x_hat, x_true), to_rmse(x_hat, x_true)]
    else:
        k_list = [3, 5, 10] if data == 'steam' else [10, 20, 50]
        scores = []
        for k in k_list:
            scores.append(to_recall(x_hat, x_true, k))
        for k in k_list:
            scores.append(to_ndcg(x_hat, x_true, k))
        return scores


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='pamap2')
    parser.add_argument('--y-ratio', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sampling', type=str2bool, default=False)

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--out', type=str, default='../out')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--emb-norm', type=str, default='unit')
    parser.add_argument('--dec-bias', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--conv', type=str, default='gcn')
    parser.add_argument('--x-loss', type=str, default='gaussian')
    parser.add_argument('--x-type', type=str, default='obs-diag')
    parser.add_argument('--feature_num', type=str, default=1)
    parser.add_argument('--node_num', type=str, default=2)
    return parser.parse_args()

def main():
    """
    Main function.
    """
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes = load_data(
        args.data, split=(0.4, 0.1, 0.5), seed=args.seed)   #In pamap2, y values are all none, we will not use that function. We have y value but we will put that in x value.

    trn_nodes = np.array(trn_nodes)
    val_nodes = np.array(val_nodes)
    test_nodes = np.array(test_nodes)

    #data = Data(x=trn_nodes, edge_index=edge_index)
    #loader = DataLoader(trn_nodes, batch_size=32)


    if is_large(args.data):  # Convert edges if a dataset is large.
        edge_index = to_edge_tensor(edge_index)

    if is_continuous(args.data):
        assert args.x_loss == 'gaussian'

    num_nodes = x_all.size(0)
    num_features = x_all.size(1)
    #num_columns = x_all.size(2)
    num_classes = int((y_all.max() + 1).item())

    x_nodes = trn_nodes
    x_features = x_all[x_nodes]
    y_nodes, y_labels = sample_y_nodes(num_nodes, y_all, args.y_ratio, args.seed)



    if args.sampling:
        model = StochasticSVGA(
            edge_index, num_nodes, num_features, num_classes, args.hidden_size,
            args.lamda, args.beta, args.layers, args.conv, args.dropout,
            args.x_type, args.x_loss, args.emb_norm, x_nodes, x_features,
            args.dec_bias)
    else:
        model = SVGA(
            edge_index, num_nodes, num_features, num_classes, args.hidden_size,
            args.lamda, args.beta, args.layers, args.conv, args.dropout,
            args.x_type, args.x_loss, args.emb_norm, x_nodes, x_features,
            args.dec_bias) #num_columns

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = to_device("cuda" if torch.cuda.is_available else "cpu")
    edge_index = edge_index.to(device)
    model = model.to(device)
    x_nodes = torch.as_tensor(x_nodes).to(device)
    x_all = torch.as_tensor(x_all).to(device)
    x_features = torch.as_tensor(x_features).to(device)

    if y_nodes is not None:
        y_nodes = y_nodes.to(device)
        y_labels = y_labels.to(device)

    def update_model(step):
        model.train()
        losses = model.to_losses(edge_index, x_nodes, x_features, y_nodes, y_labels)
        if step:
            optimizer.zero_grad()
            sum(losses).backward()
            optimizer.step()
        return tuple(l.item() for l in losses)

    @torch.no_grad()
    def evaluate_model():
        model.eval()
        x_hat_, _ = model(edge_index)
        out_list = []
        for nodes in [x_nodes, val_nodes, test_nodes]:
            if is_continuous(args.data):
                score = to_rmse(x_hat_[nodes], x_all[nodes])
            elif (args.data == 'steam'):
                score = to_recall(x_hat_[nodes], x_all[nodes], k=3)
            else:
                score = to_recall(x_hat_[nodes], x_all[nodes], k=10)
            out_list.append(score)
        return out_list

    def is_better(curr_acc_, best_acc_):
        if is_continuous(args.data):
            return curr_acc_ <= best_acc_
        else:
            return curr_acc_ >= best_acc_

    if not args.silent:
        print('-' * 47)
        print('epoch x_loss y_loss r_loss trn    val    test')
    #encoder_lstm = nn.LSTM(input_size=args.feature_num, hidden_size=1, num_layers=1, bidirectional=False).to(device)
    #decoder_lstm = nn.LSTM(input_size=1, hidden_size=args.feature_num, num_layers=1, bidirectional=False).to(device)

    logs = []
    saved_model, best_epoch, best_result = io.BytesIO(), 0, []
    best_acc = np.inf if is_continuous(args.data) else 0
    for epoch in range(args.epochs + 1):
        loss_list = []
        #encoded_x_features, _ = encoder_lstm(x_features)
        #encoded_x_features = encoded_x_features.reshape(-1, args.node_num)
        for _ in range(args.updates):
            loss_list = update_model(epoch > 0)
        acc_list = evaluate_model()
        curr_result = [epoch, loss_list, acc_list]

        val_acc = acc_list[1]
        if is_better(val_acc, best_acc):
            saved_model.seek(0)
            torch.save(model.state_dict(), saved_model)
            best_epoch = epoch
            best_acc = val_acc
            best_result = curr_result

        if not args.silent and epoch % (args.epochs // min(args.epochs, 20)) == 0:
            print_log(*curr_result)

        if args.save:
            trn_res = evaluate_last(args.data, model, edge_index, trn_nodes, x_all)
            test_res = evaluate_last(args.data, model, edge_index, test_nodes, x_all)
            logs.append([epoch, *trn_res, *test_res])

        if args.patience > 0 and epoch >= best_epoch + args.patience:
            break

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))

    val_res = evaluate_last(args.data, model, edge_index, val_nodes, x_all)
    test_res = evaluate_last(args.data, model, edge_index, test_nodes, x_all)

    if args.save:
        out_path = f'{args.out}/{args.data}/{args.seed}'
        os.makedirs(out_path, exist_ok=True)

        columns = ['epoch']
        if is_continuous(args.data):
            columns.extend([f'{a}_{b}'
                            for a in ['trn', 'test']
                            for b in ['r2', 'rmse']])
        else:
            columns.extend([f'{a}_{b}{c}'
                            for a in ['trn', 'test']
                            for b in ['rec', 'ndcg']
                            for c in [10, 20, 50]])
        df = pd.DataFrame(logs, columns=columns)
        df.to_csv(f'{out_path}/epochs.tsv', sep='\t', index=False)

        with torch.no_grad():
            x_hat, _ = model(edge_index)

        torch.save(model.state_dict(), os.path.join(out_path, 'model.pth'))
        torch.save(x_hat, os.path.join(out_path, 'features.pth'))
        torch.save(trn_nodes, os.path.join(out_path, 'trn_nodes.pth'))
        torch.save(val_nodes, os.path.join(out_path, 'val_node.pth'))
        torch.save(test_nodes, os.path.join(out_path, 'test_nodes.pth'))

    out_best = dict(epoch=best_epoch, val_res=val_res, test_res=test_res)
    if not args.silent:
        print('-' * 47)
        print_log(*best_result)
        print('-' * 47)
        print(json.dumps(out_best, indent=4, sort_keys=True))
    else:
        out = {arg: getattr(args, arg) for arg in vars(args)}
        out['out'] = out_best
        print(json.dumps(out))


if __name__ == '__main__':
    main()

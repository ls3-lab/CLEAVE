from typing import Optional

import torch
from torch.optim import Adam
import torch.nn as nn

from pGRACE.model import LogReg
from src.data.splits import get_split


def log_regression(z,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   dataset_name: str = "",
                   verbose: bool = False,
                   preload_split=None):
    data = dataset[0]
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = dataset[0].y.view(-1).to(test_device)
    data = data.to(test_device)
    num_classes = dataset[0].y.max().item() + 1

    if preload_split is not None:
        train_mask, val_mask, test_mask = preload_split
    else:
        train_mask, val_mask, test_mask = get_split(data, dataset_name=dataset_name, mode="train")

    train_mask = train_mask.to(test_device)
    val_mask = val_mask.to(test_device)
    test_mask = test_mask.to(test_device)

    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()

    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    train_set, train_label = z[train_mask], y[train_mask]
    val_set, val_label = z[val_mask], y[val_mask]
    test_set, test_label = z[test_mask], y[test_mask]

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(train_set)
        loss = nll_loss(f(output), train_label)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            if val_mask.any():
                # val split is available
                test_acc = evaluator.eval({
                    'y_true': test_label.view(-1, 1),
                    'y_pred': classifier(test_set).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': val_label.view(-1, 1),
                    'y_pred': classifier(val_set).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')

    return {'acc': best_test_acc, 'best_model': classifier}



class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}

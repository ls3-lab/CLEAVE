import torch

from tqdm import tqdm

try:
    from .Model import LinearClassifier
    from .Utils import get_optim
    from src.data.splits import get_split
except ImportError:
    from Model import LinearClassifier
    from Utils import get_optim
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from src.data.splits import get_split

def linear_evaluation(z, y, args, epochs, split=None, data=None, return_output=False):
    classes_num = y.max().item() + 1
    z = z.to(args.device)
    y = y.view(-1)
    y = y.to(args.device)
    feat_dim = z.size(1)
    model_name = args.model
    graph_name = args.dataset
    
    train_mask, val_mask, test_mask = get_split(data, dataset_name=getattr(args, 'dataset', ''), mode="train")

    train_set, train_label = z[train_mask], y[train_mask]
    test_set, test_label = z[test_mask], y[test_mask]
    val_set, val_label = z[val_mask], y[val_mask]
    # print('Train set device: {}'.format(train_set.device))
    classifier = LinearClassifier(feat_dim, classes_num).to(args.device)
    
    # Create a temporary args object for evaluation optimizer
    class EvalArgs:
        def __init__(self, optim, lr, weight_decay, momentum):
            self.optim = optim
            self.lr = lr
            self.weight_decay = weight_decay
            self.momentum = momentum  # Default momentum for SGD
    
    eval_args = EvalArgs(args.eval_optim, args.eval_lr, args.eval_weight_decay, args.momentum)
    optimizer = get_optim(classifier.parameters(), eval_args)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0

    # eval_bar = tqdm(range(args.eval_epoch), desc='Evaluating')
    for epoch in range(args.eval_epoch):
        classifier.train()
        optimizer.zero_grad()
        
        output = classifier(train_set)
        # print('Output class: {}, train_label_class: {}'.format(output.device, train_label.device))
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        classifier.eval()
        val_acc = ac(classifier, val_set, val_label)
        test_acc = ac(classifier, test_set, test_label)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            path = args.output_path + f"/{model_name}_{graph_name}_best_eval.pth"
            torch.save(classifier.state_dict(), path)
        
        # eval_bar.set_description("Eval epoch {}, best test acc: {:.4f}".format(epochs, best_test_acc))
        # eval_bar.set_postfix(loss=loss.item())
    # eval_bar.close()
    classifier.load_state_dict(torch.load(args.output_path + f"/{model_name}_{graph_name}_best_eval.pth"))
    classifier.eval()
    # print("best epoch: {}, best test acc: {:.4f}".format(best_epoch, best_test_acc))
    # output = classifier(test_set)
    # _, pred = torch.max(output, 1)
    # print(pred)
    # print(pred.shape)
    if return_output:
        return classifier, best_test_acc
    return "best epoch: {}, best test acc: {:.4f}".format(best_epoch, best_test_acc), best_test_acc
    
def ac(model, dataset, label):
    output = model(dataset)
    _, pred = torch.max(output, 1)
    total = label.size(0)
    correct = (pred == label).sum().item()
    return 100 * correct / total


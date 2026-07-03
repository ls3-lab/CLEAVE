import torch
from torch_geometric.data import Data
from omegaconf import DictConfig
from copy import deepcopy
from torch.optim import Adam
from src.retrievers.base import Retriever
from src.retrievers.EPAGCL.Model import MyModel 
from src.retrievers.EPAGCL.Linear_evaluation import LinearClassifier, ac
from src.retrievers.GCA.pGRACE.model import GRACE, Encoder, LogReg
from src.retrievers.GRACE.pGRACE.model import GRACE_Model, GRACE_Encoder, GRACE_LogReg
from src.retrievers.GRACE.pGRACE.utils import get_base_model, get_activation
from src.data.splits import split_data, get_split



class EPAGCLRetriever(Retriever):
    def __init__(self, cfg: DictConfig, system_cfg: DictConfig, data: Data, device, use_oracle: bool = False):
        super().__init__(cfg, system_cfg, data, device, use_oracle)
        

    def load_model(self):
        model_path = self.get_model_path()
        model = MyModel(self.cfg, self.data.num_features, self.cfg.hidden_dim, self.cfg.proj_dim, self.cfg.edge_drop_rate).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model.eval()
        self.model = model

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        embedding, _ = super().encode(x, edge_index)
        return embedding

    def class_mlp(self, z: torch.Tensor, y: torch.Tensor):
        data = self.data
        classes_num = y.max().item() + 1
        z = z.to(self.device)
        y = y.view(-1)
        y = y.to(self.device)
        feat_dim = z.size(1)

        train_mask, val_mask, test_mask = split_data(data)
        
        train_set, train_label = z[train_mask], y[train_mask]
        test_set, test_label = z[test_mask], y[test_mask]
        val_set, val_label = z[val_mask], y[val_mask]

        classifier = LinearClassifier(feat_dim, classes_num).to(self.device)
        optimizer = Adam(classifier.parameters(), 
                        lr=self.cfg.learning_rate, 
                        weight_decay=self.cfg.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0
        best_test_acc = 0
        bset_classifier = None
        for epoch in range(self.cfg.epochs):
            classifier.train()
            optimizer.zero_grad()
            output = classifier(train_set)
            loss = criterion(output, train_label)
            loss.backward()
            optimizer.step()

            classifier.eval()
            val_acc = ac(classifier, val_set, val_label)
            test_acc = ac(classifier, test_set, test_label)
            if val_acc > best_val_acc or (val_acc == best_val_acc and test_acc > best_test_acc):
                bset_classifier = deepcopy(classifier)

        bset_classifier.eval()
        self.best_classifier = bset_classifier
        return bset_classifier
    
    def get_layers(self):
        return self.model.get_layers()[:2]
        

class GCARetriever(Retriever):
    def __init__(self, cfg: DictConfig, system_cfg: DictConfig, data: Data, device, use_oracle: bool = False):
        super().__init__(cfg, system_cfg, data, device, use_oracle)


    def load_model(self):
        model_path = self.get_model_path()
        activation = torch.nn.PReLU()
        encoder = Encoder(in_channels=self.data.num_features, 
                        out_channels=self.cfg.hidden_dim,
                        activation=activation, 
                        k=self.cfg.num_layers).to(self.device)
        model = GRACE(encoder=encoder, 
                    num_hidden=self.cfg.hidden_dim, 
                    num_proj_hidden=self.cfg.num_proj_hidden, 
                    tau=self.cfg.tau).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        model.eval()
        self.model = model

    def class_mlp(self, z: torch.Tensor, y: torch.Tensor):
        data = self.data
        y = y.detach().to(self.device)
    
        num_hidden = z.size(1)
        num_classes = y.max().item() + 1

        train_mask, val_mask, test_mask = split_data(data)
        classifier = LogReg(num_hidden, num_classes).to(self.device)
        optimizer = Adam(classifier.parameters(), 
                        lr=self.cfg.learning_rate, 
                        weight_decay=self.cfg.weight_decay)

        train_set, train_label = z[train_mask], y[train_mask]
        
        f = torch.nn.LogSoftmax(dim=-1)
        nll_loss = torch.nn.NLLLoss()


        for epoch in range(self.cfg.epochs):
            classifier.train()
            optimizer.zero_grad()

            output = classifier(train_set)
            loss = nll_loss(f(output), train_label)

            loss.backward()
            optimizer.step()

        self.best_classifier = deepcopy(classifier)
        return classifier

    def get_layers(self):
        return self.model.get_layers()[:2]

class GRACERetriever(Retriever):
    def __init__(self, cfg: DictConfig, system_cfg: DictConfig, data: Data, device, use_oracle: bool = False):
        super().__init__(cfg, system_cfg, data, device, use_oracle)

    def load_model(self):
        model_path = self.get_model_path()
        activation = get_activation(self.cfg.activation)  # e.g., "prelu"
        base_model = get_base_model(self.cfg.base_model)  # e.g., "GCNConv"

        encoder = GRACE_Encoder(in_channels=self.data.num_features,
                                out_channels=self.cfg.hidden_dim,
                                activation=activation,
                                base_model=base_model,
                                k=self.cfg.num_layers).to(self.device)

        model = GRACE_Model(encoder=encoder,
                            num_hidden=self.cfg.hidden_dim,
                            num_proj_hidden=self.cfg.num_proj_hidden,
                            tau=self.cfg.tau).to(self.device)

        state = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(state, strict=False)
        model.eval()

        self.model = model.to(self.device)
        self.encoder = encoder.to(self.device)  # optional but useful

    # @torch.no_grad()
    # def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    #     """
    #     Returns node embeddings z (N x hidden_dim).
    #     """
    #     self.model.eval()
    #     return self.model(x.to(self.device), edge_index.to(self.device))

    def class_mlp(self, z: torch.Tensor, y: torch.Tensor):
        """
        Linear probing on frozen GRACE embeddings (same style as your GCARetriever).
        """
        y = y.detach().to(self.device)

        num_hidden = z.size(1)
        num_classes = int(y.max().item()) + 1

        train_mask, val_mask, test_mask = split_data(self.data)

        classifier = GRACE_LogReg(num_hidden, num_classes).to(self.device)
        optimizer = Adam(
            classifier.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        train_set, train_label = z[train_mask], y[train_mask]

        f = torch.nn.LogSoftmax(dim=-1)
        nll_loss = torch.nn.NLLLoss()

        epochs = getattr(self.cfg, "epochs", getattr(self.cfg, "num_epochs", 100))

        for _ in range(epochs):
            classifier.train()
            optimizer.zero_grad(set_to_none=True)

            out = classifier(train_set)
            loss = nll_loss(f(out), train_label)

            loss.backward()
            optimizer.step()

        self.best_classifier = deepcopy(classifier)
        return classifier
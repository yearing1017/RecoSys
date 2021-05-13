import torch
from torch import nn


class MLP(nn.Module):
    """
    Concatenate Embeddings that are then passed through a series of Dense layers
    """
    def __init__(self, n_user, n_item, layers, dropouts):
        super(MLP, self).__init__()

        self.layers = layers
        self.n_layers = len(layers)
        self.dropouts = dropouts
        self.n_user = n_user
        self.n_item = n_item

        self.embeddings_user = nn.Embedding(n_user, int(layers[0]/2))
        self.embeddings_item = nn.Embedding(n_item, int(layers[0]/2))

        self.mlp = nn.Sequential()
        for i in range(1,self.n_layers):
            self.mlp.add_module("linear%d" %i, nn.Linear(layers[i-1],layers[i]))
            self.mlp.add_module("relu%d" %i, torch.nn.ReLU())
            self.mlp.add_module("dropout%d" %i , torch.nn.Dropout(p=dropouts[i-1]))

        self.out = nn.Linear(in_features=layers[-1], out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

    def forward(self, users, items):

        user_emb = self.embeddings_user(users)
        item_emb = self.embeddings_item(items)
        emb_vector = torch.cat([user_emb,item_emb], dim=1)
        emb_vector = self.mlp(emb_vector)
        # preds = torch.sigmoid(self.out(emb_vector))
        preds = self.out(emb_vector)
        return preds
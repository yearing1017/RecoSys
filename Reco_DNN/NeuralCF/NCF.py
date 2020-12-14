import torch
from torch import nn
from MLP import MLP
from GMF import GMF

class NeuMF(nn.Module):
    def __init__(self, n_user, n_item, n_emb, layers, dropouts):
        super(NeuMF, self).__init__()

        self.layers = layers
        self.n_layers = len(layers)
        self.dropouts = dropouts
        self.n_user = n_user
        self.n_item = n_item

        self.mf_embeddings_user = nn.Embedding(n_user, n_emb)
        self.mf_embeddings_item = nn.Embedding(n_item, n_emb)

        self.mlp_embeddings_user = nn.Embedding(n_user, int(layers[0]/2))
        self.mlp_embeddings_item = nn.Embedding(n_item, int(layers[0]/2))
        self.mlp = nn.Sequential()
        for i in range(1,self.n_layers):
            self.mlp.add_module("linear%d" %i, nn.Linear(layers[i-1],layers[i]))
            self.mlp.add_module("relu%d" %i, torch.nn.ReLU())
            self.mlp.add_module("dropout%d" %i , torch.nn.Dropout(p=dropouts[i-1]))

        self.out = nn.Linear(in_features=n_emb+layers[-1], out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

    def forward(self, users, items):

        mf_user_emb = self.mf_embeddings_user(users)
        mf_item_emb = self.mf_embeddings_item(items)

        mlp_user_emb = self.mlp_embeddings_user(users)
        mlp_item_emb = self.mlp_embeddings_item(items)

        mf_emb_vector = mf_user_emb*mf_item_emb
        mlp_emb_vector = torch.cat([mlp_user_emb,mlp_item_emb], dim=1)
        mlp_emb_vector = self.mlp(mlp_emb_vector)

        emb_vector = torch.cat([mf_emb_vector,mlp_emb_vector], dim=1)
        # preds = torch.sigmoid(self.out(emb_vector))
        preds = self.out(emb_vector)

        return preds
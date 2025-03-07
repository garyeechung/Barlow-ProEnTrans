import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, Sequential, Sigmoid, Softmax, TransformerEncoderLayer

from .transformer import TransformerEncoderSkipPreserve


class ProEnTrans(Module):
    def __init__(self, sam, d_model=256, nhead=8, num_layers=4,
                 residual_connection=False, preserve_embedding=False):
        super().__init__()
        self.d_model = d_model
        self.layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = TransformerEncoderSkipPreserve(self.layer, num_layers=num_layers)
        self.softmax = Softmax(dim=1)
        self.prompt_encoder = sam.prompt_encoder
        self.residual_connection = residual_connection
        self.preserve_embedding = preserve_embedding

    def forward(self, samples):
        nb_points = [sample["point_coords"].shape[1] for sample in samples]
        max_nb_points = max(nb_points)
        sparse_embeddings_all = []
        padding_masks = []
        for sample in samples:
            points = sample["point_coords"], sample["point_labels"]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=None, masks=None)
            len_sparse = sparse_embeddings.shape[1]
            sparse_embeddings = F.pad(sparse_embeddings,
                                      (0, 0, 0, max_nb_points + 2 - sparse_embeddings.shape[1]),
                                      value=0)
            sparse_embeddings_all.append(sparse_embeddings)

            padding_mask = torch.zeros_like(sparse_embeddings[..., 0])
            padding_mask[:, len_sparse:-1] = 1
            padding_masks.append(padding_mask)
        sparse_embeddings_all = torch.cat(sparse_embeddings_all, dim=0)
        padding_masks = torch.cat(padding_masks, dim=0)

        if self.preserve_embedding:
            src_key_preserve_mask = 1 - padding_masks.unsqueeze(-1).repeat(1, 1, self.d_model)
            src_key_preserve_mask[:, -1] = 0
        else:
            src_key_preserve_mask = None
        sparse_embeddings_all = self.encoder(sparse_embeddings_all,
                                             src_key_padding_mask=padding_masks,
                                             residual_connection=self.residual_connection,
                                             src_key_preserve_mask=src_key_preserve_mask)
        # if residual_connection=False, src_key_preserve_mask will be ignored

        class_embeddings = sparse_embeddings_all[:, -1, :]
        sparse_embeddings_all = [sparse_embeddings_all[i, :nb_points[i] + 1] for i in range(len(samples))]

        return class_embeddings, sparse_embeddings_all


class SidesMultiLabelClassifier(Module):
    def __init__(self, sam, d_model=256, nhead=8, num_layers=4,
                 num_mlp_neurons=[256, 128, 64, 32], n_classes=2,
                 residual_connection=False, preserve_embedding=False):
        super().__init__()
        self.proentrans = ProEnTrans(sam, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                     residual_connection=residual_connection,
                                     preserve_embedding=preserve_embedding)
        self.mlp = Sequential()
        self.mlp.add_module("linear_0", Linear(d_model, num_mlp_neurons[0]))
        self.mlp.add_module("relu_0", torch.nn.ReLU())
        for i in range(1, len(num_mlp_neurons)):
            self.mlp.add_module(f"linear_{i}", Linear(num_mlp_neurons[i - 1], num_mlp_neurons[i]))
            self.mlp.add_module(f"relu_{i}", torch.nn.ReLU())
        self.mlp.add_module("linear_out", Linear(num_mlp_neurons[-1], n_classes))
        self.mlp.add_module("sigmoid", Sigmoid())

    def forward(self, samples):
        class_embeddings, sparse_embeddings_all = self.proentrans(samples)
        class_predictions = self.mlp(class_embeddings)
        return class_predictions


class BarlowTwinsCosineSimilarity(Module):
    def __init__(self, sam, d_model=256, nhead=8, num_layers=4, nb_copies=5,
                 residual_connection=False, preserve_embedding=False):
        super().__init__()
        self.proentrans = ProEnTrans(sam, d_model=d_model, nhead=nhead,
                                     num_layers=num_layers,
                                     residual_connection=residual_connection,
                                     preserve_embedding=preserve_embedding)
        self.nb_copies = nb_copies
        self.softmax = Softmax(dim=1)

    def forward(self, samples):
        class_embeddings, sparse_embeddings_all = self.proentrans(samples)
        cos_sims = self.compute_cos_sims(class_embeddings)
        return cos_sims

    def compute_cos_sims(self, class_embeddings):
        cls_emb_softmax = self.softmax(class_embeddings)
        cos_sims = []
        for i in range(cls_emb_softmax.shape[0]):
            for j in range(i, cls_emb_softmax.shape[0]):
                cos_sim = F.cosine_similarity(cls_emb_softmax[i], cls_emb_softmax[j], dim=0)
                cos_sims.append(cos_sim)
        cos_sims = torch.stack(cos_sims)

        return cos_sims

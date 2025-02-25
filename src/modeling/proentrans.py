import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, Sequential, Sigmoid, Softmax, TransformerEncoder, TransformerEncoderLayer


class ProEnTrans(Module):
    def __init__(self, sam, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = TransformerEncoder(self.layer, num_layers=num_layers)
        self.softmax = Softmax(dim=1)
        self.prompt_encoder = sam.prompt_encoder

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

        sparse_embeddings_all = self.encoder(sparse_embeddings_all, src_key_padding_mask=padding_masks)
        class_embeddings = sparse_embeddings_all[:, -1, :]
        sparse_embeddings_all = [sparse_embeddings_all[i, :nb_points[i] + 1] for i in range(len(samples))]

        return class_embeddings, sparse_embeddings_all


class SidesMultiLabelClassifier(Module):
    def __init__(self, sam, d_model=256, nhead=8, num_layers=4,
                 num_mlp_neurons=[256, 128, 64, 32], n_classes=2):
        super().__init__()
        self.proentrans = ProEnTrans(sam, d_model=d_model, nhead=nhead, num_layers=num_layers)
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
    def __init__(self, sam, d_model=256, nhead=8, num_layers=4, nb_copies=5):
        super().__init__()
        self.proentrans = ProEnTrans(sam, d_model=d_model, nhead=nhead, num_layers=num_layers)
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

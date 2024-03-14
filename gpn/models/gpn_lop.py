from typing import Dict, Tuple, List
from sklearn import mixture
from sklearn.random_projection import SparseRandomProjection
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.utils as tu
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from gpn.layers.appnp_propagation import APPNPPropagation
from gpn.nn import mixture_uce_loss, entropy_reg, categorical_entropy_reg, expected_categorical_entropy
from gpn.utils import apply_mask
from gpn.utils import Prediction
from .gpn_base import GPN


class GPN_LOP(GPN):
    """Graph Posterior Network model using linear opinion pooling instead of alpha parameter pooling, i.e., using a mixture of Dirichlet distributions."""

    default_normalization = "rw"
    default_x_prune_threshold = None

    def forward(self, data):
        N = data.x.size(0)
        if data.edge_index is not None:
            adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(N, N))
        else:
            adj_t = data.adj_t
        h = self.input_encoder(data.x)
        z = self.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.get_class_probalities(data)
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if (
            isinstance(self.params.alpha_evidence_scale, str)
            and "-plus-classes" in self.params.alpha_evidence_scale
        ):
            further_scale = self.params.num_classes
        else:
            further_scale = 1.0

        log_beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.params.dim_latent, further_scale=further_scale
        )
        beta_ft = log_beta_ft.exp()

        alpha_features = 1.0 + beta_ft

        if self.params.sparse_propagation:
            I = SparseTensor.eye(N, dtype=torch.float32, device=data.x.device)  # type: ignore
        else:
            I = torch.eye(N, dtype=torch.float32, device=data.x.device)

        propagation_weights: torch.Tensor | SparseTensor = self.propagation(I, adj_t)
        evidence_ft = beta_ft.sum(-1)
        evidence = propagation_weights @ evidence_ft.view(-1, 1)
        alpha = propagation_weights @ alpha_features

        soft_ft = alpha_features / alpha_features.sum(-1, keepdim=True)
        soft = propagation_weights @ soft_ft
        log_soft = soft.log()
        max_soft, hard = soft.max(dim=-1)

        epistemic_entropy_diff: None | torch.Tensor

        if self.training:
            fo_neg_entropy = None
            exp_fo_neg_entropy = None
            epistemic_entropy_diff = None
        else:
            fo_neg_entropy = categorical_entropy_reg(soft, 1, reduction="none")
            exp_fo_neg_entropy_ft = -expected_categorical_entropy(alpha_features)
            exp_fo_neg_entropy = (propagation_weights @ exp_fo_neg_entropy_ft.view(-1, 1)).view(-1)

            epistemic_entropy_diff = fo_neg_entropy - exp_fo_neg_entropy

        neg_entropy_features = entropy_reg(
            alpha_features, 1, approximate=True, reduction="none"
        )
        so_neg_entropy = (propagation_weights @ neg_entropy_features.view(-1, 1)).view(-1)
        so_neg_entropy = so_neg_entropy + categorical_entropy_reg(
            propagation_weights, 1, reduction="none"
        )

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha,  # this alpha is not meaningful, since LOP produces a mixture of Dirichlet distributions
            alpha_features=alpha_features,
            propagation_weights=propagation_weights,
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            logits=None,
            latent=z,
            latent_features=z,
            hidden=h,
            hidden_features=h,
            evidence=evidence,
            evidence_ft=evidence_ft,
            log_ft_per_class=log_q_ft_per_class,
            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,
            # sample confidence scores
            sample_confidence_total=max_soft,
            sample_confidence_total_entropy=fo_neg_entropy,
            sample_confidence_aleatoric=max_soft,
            sample_confidence_aleatoric_entropy=exp_fo_neg_entropy,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_epistemic_entropy=so_neg_entropy,
            sample_confidence_epistemic_entropy_diff=epistemic_entropy_diff,
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

    def uce_loss(
        self, prediction: Prediction, data: Data, approximate=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_features = prediction.alpha_features
        masked_pred: dict[str, torch.Tensor]
        masked_pred, y = apply_mask(
            data,
            dict(
                mixture_weights=prediction.propagation_weights,
                neg_entropy=prediction.sample_confidence_epistemic_entropy,
            ),
            split="train",
        )  # type: ignore
        mixture_weights = masked_pred["mixture_weights"]
        neg_entropy = masked_pred["neg_entropy"]
        reg = self.params.entropy_reg

        uce = mixture_uce_loss(alpha_features, mixture_weights, y, reduction="sum")
        reg_loss = reg * neg_entropy.sum()
        return uce, reg_loss

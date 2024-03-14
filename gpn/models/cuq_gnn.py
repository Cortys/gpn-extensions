import torch
import torch.nn as nn
from torch_geometric.data import Data
from gpn.layers.evidence import Density
from gpn.models.gat import GAT
from gpn.models.gcn import GCN
from gpn.models.gpn_base import GPN
from gpn.nn.loss import categorical_entropy_reg, entropy_reg, expected_categorical_entropy
from gpn.utils.config import ModelConfiguration
from gpn.utils.prediction import Prediction


class CUQ_GNN(GPN):
    """Commitee-based Uncertainty Quantification Graph Neural Network"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)

    def init_input_encoder(self):
        pass

    def init_propagation(self):
        assert self.params.convolution_name in (
            "gcn",
            "gat",
        )
        assert isinstance(self.params.dim_hidden, int)
        params = self.params.clone()
        params.set_values(
            num_classes=self.params.dim_hidden,
        )
        if self.params.convolution_name == "gcn":
            self.gnn = GCN(params)
        else:
            self.gnn = GAT(params)

    def forward_impl(self, data: Data) -> Prediction:
        h = self.gnn(data)
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

        beta = self.evidence(
            log_q_ft_per_class, dim=self.params.dim_latent, further_scale=further_scale
        ).exp()

        alpha = 1.0 + beta

        soft = alpha / alpha.sum(-1, keepdim=True)
        logits = None
        log_soft = soft.log()
        max_soft, hard = soft.max(dim=-1)

        if self.training:
            fo_neg_entropy = None
            exp_fo_neg_entropy = None
            epistemic_entropy_diff = None
        else:
            fo_neg_entropy = categorical_entropy_reg(soft, 1, reduction="none")
            exp_fo_neg_entropy = -expected_categorical_entropy(alpha)
            epistemic_entropy_diff = fo_neg_entropy - exp_fo_neg_entropy
        so_neg_entropy = entropy_reg(
            alpha, 1, approximate=True, reduction="none"
        )

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha,
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            logits=logits,
            latent=z,
            latent_features=z,
            hidden=h,
            hidden_features=h,
            evidence=beta.sum(-1),
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
            sample_confidence_structure=None,
        )
        # ---------------------------------------------------------------------------------

        return pred

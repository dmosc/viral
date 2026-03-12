import torch
import torch.nn as nn

from transformers import AutoModel

from src.config import Config


class ViralityPredictor(nn.Module):
    def __init__(self, config: Config, tabular_means=None, tabular_stds=None):
        super().__init__()
        self.config = config
        self.register_buffer('tabular_means',
                             tabular_means if tabular_means is not None else torch.zeros(config.num_tabular_features))
        self.register_buffer('tabular_stds',
                             tabular_stds if tabular_stds is not None else torch.ones(config.num_tabular_features))
        self.text_model = AutoModel.from_pretrained(self.config.text_model_id)
        self.video_model = AutoModel.from_pretrained(
            self.config.video_model_id)
        # Freeze the pretrained models
        self.text_model.requires_grad_(False)
        self.video_model.requires_grad_(False)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.config.num_tabular_features, self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.ReLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.Dropout(self.config.dropout)
        )
        # DistilBERT hidden size + VideoMAE hidden size + tabular MLP
        self.late_fusion = self.text_model.config.hidden_size + \
            self.video_model.config.hidden_size + self.config.d_model
        self.late_fusion_mlp = nn.Sequential(
            nn.Linear(self.late_fusion, self.config.d_model),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        self.engagement_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.config.d_model // 2, 1),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.config.d_model // 2, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.config.d_model // 2, 1)
        )
        self.regression_loss = nn.HuberLoss(reduction='none')
        self.register_buffer("pos_weight", torch.tensor(
            [config.viral_loss_weight]))
        self.classification_loss = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pixel_values: torch.Tensor, tabular_features: torch.Tensor,
                labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # Text: Extract [CLS] token (index 0)
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        # Video: Global average pool across spatial-temporal tokens
        video_output = self.video_model(
            pixel_values=pixel_values
        ).last_hidden_state.mean(dim=1)
        # Tabular: Run tabular features through the tabular MLP.
        tabular_output = self.tabular_mlp(tabular_features)
        # Perform a late fusion of all branches.
        late_fusion = torch.cat(
            [text_output, video_output, tabular_output], dim=-1
        )
        shared_features = self.late_fusion_mlp(late_fusion)
        engagement_logits = self.engagement_head(shared_features)
        velocity_logits = self.velocity_head(shared_features)
        viral_logits = self.classification_head(shared_features)
        output = {
            "regression_logits": torch.cat([engagement_logits, velocity_logits],
                                           dim=-1),
            "classification_logits": viral_logits
        }
        if labels is not None:
            regression_targets = labels[:, :2]
            is_viral_target = labels[:, 2].view(-1, 1)
            loss_reg = self.regression_loss(
                output["regression_logits"], regression_targets).mean()
            loss_cls = self.classification_loss(viral_logits, is_viral_target)
            output["loss"] = (loss_reg * self.config.regression_loss_contribution) + \
                             (loss_cls * self.config.classification_loss_contribution)
        return output

    @torch.no_grad()
    def predict_scores(self, **kwargs):
        self.eval()
        output = self.forward(**kwargs)
        regression_preds = torch.expm1(output['regression_logits'])
        viral_prob = torch.sigmoid(output['classification_logits'])
        return {
            "engagement": regression_preds[:, 0],
            "velocity": regression_preds[:, 1],
            "viral_prob": viral_prob.squeeze(-1)
        }

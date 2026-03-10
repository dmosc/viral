import torch
import torch.nn as nn

from transformers import AutoModel

from src.config import Config


class ViralityPredictor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.text_model = AutoModel.from_pretrained(self.config.text_model_id)
        self.video_model = AutoModel.from_pretrained(
            self.config.video_model_id)
        # Freeze the pretrained models
        self.text_model.requires_grad_(False)
        self.video_model.requires_grad_(False)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(config.num_tabular_features, self.config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
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
            nn.Softplus()
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.config.d_model // 2, 1),
            nn.Softplus()
        )
        self.loss = nn.HuberLoss(reduction='none')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pixel_values: torch.Tensor, tabular_features: torch.Tensor,
                labels: torch.Tensor | None) -> dict[str, torch.Tensor]:
        # Text: Extract [CLS] token (index 0)
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        # Video: Global average pool across spatial-temporal tokens
        video_output = self.video_model(
            pixel_values=pixel_values
        ).last_hidden_state.mean(dim=1)
        tabular_output = self.tabular_mlp(tabular_features)
        late_fusion = torch.cat(
            [text_output, video_output, tabular_output], dim=-1
        )
        shared_features = self.late_fusion_mlp(late_fusion)
        eng_pred = self.engagement_head(shared_features)
        vel_pred = self.velocity_head(shared_features)
        logits = torch.cat([eng_pred, vel_pred], dim=-1)
        output = {"logits": logits}
        if labels is not None:
            targets, is_viral = labels[:, :2], labels[:, 2]
            weights = 1.0 + (self.config.viral_loss_weight - 1.0) * is_viral
            base_loss = self.loss(logits, targets).mean(dim=1)
            output["loss"] = (weights * base_loss).mean()
        return output

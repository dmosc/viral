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
        self.tabular_mlp = nn.Sequential(
            nn.Linear(config.num_tabular_features, self.config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        # DistilBERT hidden size + VideoMAE hidden size + tabular MLP
        self.late_fusion_1 = self.text_model.config.hidden_size + \
            self.video_model.config.hidden_size + self.config.d_model
        self.classifier = nn.Sequential(
            nn.Linear(self.late_fusion_1, self.config.d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.config.d_model, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pixel_values: torch.Tensor, tabular_features: torch.Tensor,
                labels: torch.Tensor | None) -> dict[str, torch.Tensor]:
        # Text: Extract [CLS] token (index 0)
        text_output = self.text_model(input_ids=input_ids,
                                      attention_mask=attention_mask).last_hidden_state[:, 0, :]
        # Video: Global average pool across spatial-temporal tokens
        video_output = self.video_model(
            pixel_values=pixel_values).last_hidden_state.mean(dim=1)
        tabular_output = self.tabular_mlp(tabular_features)
        late_fusion = torch.cat(
            [text_output, video_output, tabular_output], dim=-1)
        logits = self.classifier(late_fusion)
        output = {"logits": logits}
        if labels is not None:
            output["loss"] = self.loss(logits, labels)
        return output

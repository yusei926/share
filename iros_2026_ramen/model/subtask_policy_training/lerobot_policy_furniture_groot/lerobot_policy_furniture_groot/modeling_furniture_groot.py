"""GR00T N1.7 action policy with an independent diagnostic progress head."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from huggingface_hub import snapshot_download

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import get_device_from_parameters

from .configuration_furniture_groot import FurnitureGrootConfig
from .processor_furniture_groot import PROGRESS_TARGET_KEY, PROGRESS_VALID_KEY


class FurnitureGrootPolicy(GrootPolicy):
    """Preserve the official action model and add only an auxiliary progress loss."""

    name = "furniture_groot"
    config_class = FurnitureGrootConfig

    def __init__(self, config: FurnitureGrootConfig, **kwargs):
        super().__init__(config, **kwargs)
        model_config = self._groot_model.config
        input_dim = int(model_config.backbone_embedding_dim) + int(model_config.input_embedding_dim)
        self.progress_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.progress_hidden_dim),
            nn.GELU(),
            nn.Linear(config.progress_hidden_dim, config.chunk_size),
        )
        if not config.progress_enabled:
            self.progress_head.requires_grad_(False)

    def _create_groot_model(self):
        """Resolve a remote base model to the pinned revision before loading it."""
        original_path = self.config.base_model_path
        if original_path is None or Path(original_path).expanduser().exists():
            return super()._create_groot_model()

        pinned_path = snapshot_download(
            repo_id=original_path,
            repo_type="model",
            revision=self.config.base_model_revision,
        )
        self.config.base_model_path = pinned_path
        try:
            return super()._create_groot_model()
        finally:
            self.config.base_model_path = original_path

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        prepared = dict(batch)
        action_mask = prepared.get("action_mask")
        if not isinstance(action_mask, Tensor) or action_mask.ndim != 3:
            raise ValueError("Furniture-GR00T training requires action_mask with shape (B,H,132)")
        action_mask = action_mask.clone()
        action_mask[..., self.config.valid_action_dim :] = 0
        prepared["action_mask"] = action_mask
        groot_inputs = self._filter_groot_inputs(prepared, include_action=True)
        device = get_device_from_parameters(self)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            outputs = self._groot_model.forward(groot_inputs)
            action_loss = outputs.get("loss")
            if action_loss is None:
                raise RuntimeError("GR00T action model did not return loss")
            if self.config.progress_enabled:
                progress_prediction = self._predict_progress_from_features(
                    outputs["backbone_features"],
                    outputs["state_features"],
                    attention_mask=groot_inputs.get("attention_mask"),
                )
                progress_loss, monotonicity_loss = self._progress_losses(
                    progress_prediction,
                    prepared.get(PROGRESS_TARGET_KEY),
                    prepared.get(PROGRESS_VALID_KEY),
                )
            else:
                progress_loss = action_loss * 0.0
                monotonicity_loss = action_loss * 0.0
            total_loss = (
                action_loss
                + self.config.progress_loss_weight * progress_loss
                + self.config.progress_monotonicity_weight * monotonicity_loss
            )

        return total_loss, {
            "loss": float(total_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "progress_loss": float(progress_loss.detach()),
            "progress_monotonicity_loss": float(monotonicity_loss.detach()),
            "valid_action_fraction": float(action_mask.mean().detach()),
        }

    def _predict_progress_from_features(
        self,
        backbone_features: Tensor,
        state_features: Tensor,
        *,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if attention_mask is not None and attention_mask.shape[:2] == backbone_features.shape[:2]:
            mask = attention_mask.to(backbone_features.dtype).unsqueeze(-1)
            pooled_backbone = (backbone_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            pooled_backbone = backbone_features.mean(dim=1)
        pooled_state = state_features.mean(dim=1)
        features = torch.cat((pooled_backbone, pooled_state), dim=-1)
        return torch.sigmoid(
            self.progress_head(features).to(torch.float32)
        ).unsqueeze(-1)

    def _progress_losses(
        self,
        prediction: Tensor,
        target: Tensor | None,
        valid: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        zero = prediction.sum() * 0.0
        if not self.config.progress_enabled:
            return zero, zero
        if not isinstance(target, Tensor) or not isinstance(valid, Tensor):
            raise ValueError("progress-enabled training requires progress_target and progress_valid")
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        valid = valid.to(device=prediction.device, dtype=torch.bool)
        if target.shape != prediction.shape or valid.shape != prediction.shape:
            raise ValueError(
                "progress target/mask must match prediction "
                f"{tuple(prediction.shape)}, got {tuple(target.shape)} and {tuple(valid.shape)}"
            )
        if not bool(valid.any()):
            return zero, zero
        pointwise = functional.smooth_l1_loss(prediction, target, reduction="none")
        progress_loss = pointwise[valid].mean()
        pair_valid = valid[:, :-1] & valid[:, 1:]
        if bool(pair_valid.any()):
            decreases = functional.relu(
                prediction[:, :-1, :] - prediction[:, 1:, :]
            )
            monotonicity_loss = decreases[pair_valid].mean()
        else:
            monotonicity_loss = zero
        return progress_loss, monotonicity_loss

    @torch.no_grad()
    def predict_progress(self, batch: dict[str, Tensor]) -> Tensor:
        """Return diagnostic progress without changing action generation."""
        self.eval()
        groot_inputs = self._filter_groot_inputs(batch, include_action=False)
        backbone_inputs, action_inputs = self._groot_model.prepare_input(groot_inputs)
        device = get_device_from_parameters(self)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            backbone_outputs = self._groot_model.backbone(backbone_inputs)
            features = self._groot_model.action_head._encode_features(
                backbone_outputs, action_inputs
            )
        return self._predict_progress_from_features(
            features["backbone_features"],
            features["state_features"],
            attention_mask=groot_inputs.get("attention_mask"),
        )

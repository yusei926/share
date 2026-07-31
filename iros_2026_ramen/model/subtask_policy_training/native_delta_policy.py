"""Native models and data utilities for the real-only flip-table benchmarks.

The LeRobot 0.6 ACT implementation intentionally supports one observation frame
and shares one visual backbone across cameras.  The benchmark in this repository
requires two frames and a separate ResNet-18 for every physical camera, so this
module keeps that small extension local and explicit rather than silently
changing the requested configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM = 19
ACTION_DIM = 16
OBS_STEPS = 2
CHUNK_SIZE = 16
# MP4's 30 Hz timebase and parquet float timestamps differ by up to 0.2 ms in
# this recorded dataset.  One millisecond remains far below a video frame.
VIDEO_TIMESTAMP_TOLERANCE_S = 1e-3
# The local cache has GOP=8, which makes TorchCodec random access fast enough
# for H100 training.  The runner bounds cache ownership by limiting workers.
VIDEO_BACKEND = "torchcodec"


def configure_serial_video_decode() -> None:
    """Use bounded, serial RGB decoding in LeRobot DataLoader workers.

    The source dataset has three independent MP4 streams.  LeRobot normally
    decodes those streams concurrently inside every worker.  TorchCodec's
    one-entry decoder cache is intentionally used here to bound host memory on
    long videos, but concurrent cache eviction can close a stream while a
    sibling decode is seeking it.  This process-local override serializes the
    three streams and preserves LeRobot's depth handling for completeness.
    """
    from lerobot.datasets.dataset_reader import DatasetReader, dequantize_depth
    from lerobot.datasets.video_utils import decode_video_frames

    if getattr(DatasetReader, "_flip_table_serial_video_decode", False):
        return

    def query_videos_serial(self: Any, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, Tensor]:
        episode = self._meta.episodes[ep_idx]
        result: dict[str, Tensor] = {}
        for video_key, timestamps in query_timestamps.items():
            from_timestamp = episode[f"videos/{video_key}/from_timestamp"]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, video_key)
            frames = decode_video_frames(
                video_path,
                [from_timestamp + timestamp for timestamp in timestamps],
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
                is_depth=video_key in self._meta.depth_keys,
            )
            if video_key in self._meta.depth_keys:
                depth_encoder = self._depth_encoder_configs[video_key]
                frames = dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            result[video_key] = frames.squeeze(0)
        return result

    DatasetReader._query_videos = query_videos_serial
    DatasetReader._flip_table_serial_video_decode = True


@dataclass(frozen=True)
class NativeACTConfig:
    """Serialized architecture contract for the custom ACT checkpoint."""

    type: str = "flip_table_native_act_chunk_relative"
    observation_horizon: int = OBS_STEPS
    action_horizon: int = CHUNK_SIZE
    action_execution_steps: int = 8
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    dim_model: int = 512
    n_encoder_layers: int = 4
    n_decoder_layers: int = 7
    n_heads: int = 8
    latent_dim: int = 32
    vision_backbone: str = "resnet18"
    separate_camera_encoders: bool = True
    image_size: tuple[int, int] = (240, 320)


def _resnet18_feature_extractor(*, pretrained: bool) -> tuple[nn.Module, int]:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
    return IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"}), backbone.fc.in_features


class NativeACTDeltaPolicy(nn.Module):
    """ACT with two observation frames and one ResNet-18 per camera role."""

    def __init__(self, config: NativeACTConfig = NativeACTConfig(), *, pretrained_backbones: bool = True):
        super().__init__()
        self.config = config
        self.camera_backbones = nn.ModuleDict()
        feature_dim: int | None = None
        for index, _ in enumerate(CAMERA_KEYS):
            backbone, dimensions = _resnet18_feature_extractor(pretrained=pretrained_backbones)
            self.camera_backbones[str(index)] = backbone
            feature_dim = dimensions
        assert feature_dim is not None
        self.image_projection = nn.Conv2d(feature_dim, config.dim_model, kernel_size=1)
        self.image_positional = nn.Parameter(torch.zeros(OBS_STEPS, len(CAMERA_KEYS), 1, config.dim_model))
        # A 240x320 input yields an 8x10 ResNet-18 layer4 feature grid.  Without
        # these spatial positions, the transformer cannot distinguish locations
        # after flattening the grid into visual tokens.
        self.image_spatial_positional = nn.Parameter(torch.zeros(1, 8 * 10, config.dim_model))
        self.state_projection = nn.Linear(config.state_dim, config.dim_model)
        self.state_positional = nn.Parameter(torch.zeros(OBS_STEPS, 1, config.dim_model))

        self.latent_token = nn.Parameter(torch.zeros(1, 1, config.dim_model))
        self.latent_projection = nn.Linear(config.latent_dim, config.dim_model)
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, config.dim_model))
        self.vae_state_projection = nn.Linear(config.observation_horizon * config.state_dim, config.dim_model)
        self.vae_action_projection = nn.Linear(config.action_dim, config.dim_model)
        self.vae_position = nn.Parameter(torch.zeros(1, 2 + config.action_horizon, config.dim_model))
        vae_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.vae_encoder = nn.TransformerEncoder(vae_layer, num_layers=config.n_encoder_layers)
        self.vae_distribution = nn.Linear(config.dim_model, config.latent_dim * 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
        self.decoder_queries = nn.Parameter(torch.zeros(1, config.action_horizon, config.dim_model))
        self.action_head = nn.Linear(config.dim_model, config.action_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in (
            self.image_positional,
            self.image_spatial_positional,
            self.state_positional,
            self.latent_token,
            self.vae_cls,
            self.vae_position,
            self.decoder_queries,
        ):
            nn.init.normal_(parameter, std=0.02)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze all camera encoders without changing the model contract."""
        for parameter in self.camera_backbones.parameters():
            parameter.requires_grad = trainable

    def set_backbone_final_stages_trainable(self) -> None:
        """Fine-tune layer3/layer4 only, retaining frozen early visual features."""
        self.set_backbone_trainable(False)
        for backbone in self.camera_backbones.values():
            for name, parameter in backbone.named_parameters():
                if name.startswith("layer3") or name.startswith("layer4"):
                    parameter.requires_grad = True

    def _image_tokens(self, images: Tensor) -> Tensor:
        # images: [B, T=2, C=3, 3, H, W]
        if images.ndim != 6 or images.shape[1:3] != (OBS_STEPS, len(CAMERA_KEYS)):
            raise ValueError(f"images must be [B,2,3,3,H,W], got {tuple(images.shape)}")
        tokens: list[Tensor] = []
        for time_index in range(OBS_STEPS):
            for camera_index in range(len(CAMERA_KEYS)):
                features = self.camera_backbones[str(camera_index)](images[:, time_index, camera_index])["feature_map"]
                projected = self.image_projection(features)
                flattened = projected.flatten(2).transpose(1, 2)
                if flattened.shape[1] != self.image_spatial_positional.shape[1]:
                    raise ValueError(
                        "unexpected visual feature grid; expected 8x10 tokens for 240x320 input, "
                        f"got {flattened.shape[1]}"
                    )
                tokens.append(
                    flattened
                    + self.image_positional[time_index, camera_index]
                    + self.image_spatial_positional
                )
        return torch.cat(tokens, dim=1)

    def _encode_context(self, images: Tensor, state: Tensor, latent: Tensor) -> Tensor:
        if state.shape[1:] != (OBS_STEPS, self.config.state_dim):
            raise ValueError(f"state must be [B,2,{self.config.state_dim}], got {tuple(state.shape)}")
        state_tokens = self.state_projection(state) + self.state_positional.transpose(0, 1)
        latent_token = self.latent_projection(latent).unsqueeze(1) + self.latent_token
        return self.encoder(torch.cat((latent_token, state_tokens, self._image_tokens(images)), dim=1))

    def _sample_latent(self, state: Tensor, actions: Tensor | None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch = state.shape[0]
        if actions is None:
            return torch.zeros(batch, self.config.latent_dim, device=state.device, dtype=state.dtype), None, None
        state_token = self.vae_state_projection(state.flatten(1)).unsqueeze(1)
        action_tokens = self.vae_action_projection(actions)
        tokens = torch.cat((self.vae_cls.expand(batch, -1, -1), state_token, action_tokens), dim=1)
        encoded = self.vae_encoder(tokens + self.vae_position)
        distribution = self.vae_distribution(encoded[:, 0])
        mean, log_variance = distribution.chunk(2, dim=-1)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return latent, mean, log_variance

    def forward(self, images: Tensor, state: Tensor, actions: Tensor | None = None) -> tuple[Tensor, Tensor | None, Tensor | None]:
        latent, mean, log_variance = self._sample_latent(state, actions if self.training else None)
        memory = self._encode_context(images, state, latent)
        queries = self.decoder_queries.expand(images.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded), mean, log_variance

    @torch.inference_mode()
    def predict_action_chunk(self, images: Tensor, state: Tensor) -> Tensor:
        self.eval()
        return self.forward(images, state)[0]


def normalizer_from_stats(stats: dict[str, Any], key: str, *, device: torch.device) -> tuple[Tensor, Tensor]:
    values = stats[key]
    mean = torch.tensor(values["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(values["std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return mean, std


def normalize(value: Tensor, stats: tuple[Tensor, Tensor]) -> Tensor:
    mean, std = stats
    return (value - mean) / std


def denormalize(value: Tensor, stats: tuple[Tensor, Tensor]) -> Tensor:
    mean, std = stats
    return value * std + mean


def kl_divergence(mean: Tensor | None, log_variance: Tensor | None) -> Tensor:
    if mean is None or log_variance is None:
        return torch.zeros((), device=mean.device if mean is not None else "cpu")
    # The KL term is near zero after VAE convergence.  Computing its
    # cancellation-prone form under bf16 autocast can produce a negative value,
    # despite KL(q || N(0, I)) being non-negative.  Keep this regularizer in
    # float32 and clamp only residual numerical error.
    mean_32 = mean.float()
    log_variance_32 = log_variance.float()
    per_sample = 0.5 * (
        mean_32.square() + log_variance_32.exp() - 1.0 - log_variance_32
    ).sum(dim=-1)
    return per_sample.clamp_min(0.0).mean()


def save_native_act_checkpoint(
    *,
    model: NativeACTDeltaPolicy,
    output_dir: Path,
    config: NativeACTConfig,
    training_config: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    """Write a self-contained, HF-uploadable checkpoint for the custom policy."""
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, output_dir / "model.safetensors")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    (output_dir / "normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (output_dir / name).write_text(json.dumps({"steps": []}, indent=2), encoding="utf-8")
    source = Path(__file__)
    (output_dir / "modeling_flip_table_native_act.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def load_native_act_checkpoint(model_dir: Path, *, device: torch.device) -> tuple[NativeACTDeltaPolicy, dict[str, Any]]:
    from safetensors.torch import load_file

    config = NativeACTConfig(**json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
    model = NativeACTDeltaPolicy(config, pretrained_backbones=False).to(device)
    model.load_state_dict(load_file(model_dir / "model.safetensors", device=str(device)))
    model.eval()
    stats = json.loads((model_dir / "normalization.json").read_text(encoding="utf-8"))
    return model, stats

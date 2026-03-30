"""Trajectory diffusion head for ego-vehicle planning.

Adapted from DiffusionDrive's ConditionalUnet1D with simplified conditioning:
- Global conditioning only (VLM hidden state), no BEV/agent/map cross-attention
- Truncated diffusion schedule (T_trunc=40, 2 inference steps)
- K-means anchor initialization per driving command
"""
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDIMScheduler


# ---------------------------------------------------------------------------
# UNet building blocks (from DiffusionDrive conditional_unet1d.py)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish"""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)  # (B, C, 1)
        out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1D U-Net for trajectory denoising with global conditioning.

    Input:  (B, T, 2)  noisy trajectory
    Output: (B, T, 2)  denoised trajectory
    """

    def __init__(
        self,
        input_dim: int = 2,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: list = (128, 256),
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        sample:      (B, T, input_dim)
        timestep:    (B,) or scalar
        global_cond: (B, global_cond_dim)
        returns:     (B, T, input_dim)
        """
        x = sample.permute(0, 2, 1)  # (B, C, T)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.permute(0, 2, 1)  # (B, T, C)


# ---------------------------------------------------------------------------
# Trajectory Diffusion Head wrapper
# ---------------------------------------------------------------------------

class TrajectoryDiffusionHead(nn.Module):
    """Truncated diffusion head for trajectory prediction.

    Inputs:
      1. command index (int) → selects anchors from [3, num_modes, future_steps, 2]
      2. anchor trajectories → noisy starting point
      3. global_cond (VLM hidden state projected to cond_dim) → scene context

    Training: adds noise to GT at truncated timesteps, UNet predicts clean sample.
    Inference: 2 DDIM steps from anchored noise → denoised trajectory.
    """

    def __init__(
        self,
        anchor_path: str,
        future_steps: int = 12,
        num_modes: int = 6,
        global_cond_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (128, 256),
        num_train_timesteps: int = 1000,
        t_trunc: int = 40,
        num_inference_steps: int = 2,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.t_trunc = t_trunc
        self.num_inference_steps = num_inference_steps

        # Load anchors: [3, num_modes, future_steps, 2]
        anchors = np.load(anchor_path).astype(np.float32)
        self.register_buffer("anchors", torch.from_numpy(anchors))

        # UNet
        self.unet = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
        )

        # DDIM scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

    def _select_anchors(self, command: torch.Tensor) -> torch.Tensor:
        """Select anchors based on command index.

        command: (B,) int tensor with values in {0, 1, 2}
        returns: (B, num_modes, future_steps, 2)
        """
        return self.anchors[command]

    def forward_train(
        self,
        gt_traj: torch.Tensor,
        command: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        gt_traj:     (B, future_steps, 2) ground-truth trajectory
        command:     (B,) int command index
        global_cond: (B, cond_dim) projected VLM features
        returns:     dict with 'loss' and 'pred_traj'
        """
        bs = gt_traj.shape[0]
        device = gt_traj.device

        # Select anchors for each sample's command
        anchors = self._select_anchors(command)  # (B, M, T, 2)

        # Add noise to GT trajectory (same noise level for all modes)
        timesteps = torch.randint(0, self.t_trunc, (bs,), device=device, dtype=torch.long)
        noise = torch.randn_like(gt_traj)

        # Normalize GT to [-1, 1] range for diffusion
        gt_norm = self._normalize(gt_traj)
        noisy = self.scheduler.add_noise(gt_norm, noise, timesteps)

        # Expand for all modes: replicate noisy GT across modes
        # Each mode gets the same noisy GT but different anchor context
        noisy_expanded = noisy.unsqueeze(1).expand(-1, self.num_modes, -1, -1)  # (B, M, T, 2)
        noisy_flat = noisy_expanded.reshape(bs * self.num_modes, self.future_steps, 2)

        # Expand conditioning
        cond_flat = global_cond.unsqueeze(1).expand(-1, self.num_modes, -1).reshape(bs * self.num_modes, -1)
        timesteps_flat = timesteps.unsqueeze(1).expand(-1, self.num_modes).reshape(bs * self.num_modes)

        # UNet predicts clean sample
        pred_flat = self.unet(noisy_flat, timesteps_flat, global_cond=cond_flat)
        pred = pred_flat.reshape(bs, self.num_modes, self.future_steps, 2)

        # Denormalize predictions
        pred_denorm = self._denormalize(pred)

        # Winner-take-all: find best mode per sample
        with torch.no_grad():
            per_mode_l2 = ((pred_denorm - gt_traj.unsqueeze(1)) ** 2).sum(dim=-1).mean(dim=-1)  # (B, M)
            best_mode = per_mode_l2.argmin(dim=-1)  # (B,)

        # Loss on best mode only
        best_pred = pred_denorm[torch.arange(bs, device=device), best_mode]  # (B, T, 2)
        loss = ((best_pred - gt_traj) ** 2).sum(dim=-1).mean()

        return {"loss": loss, "pred_traj": best_pred.detach()}

    @torch.inference_mode()
    def forward_inference(
        self,
        command: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: denoise from anchored Gaussian in 2 DDIM steps.

        command:     (B,) int command index
        global_cond: (B, cond_dim) projected VLM features
        returns:     (B, future_steps, 2) best trajectory
        """
        bs = command.shape[0]
        device = command.device

        # Select anchors and normalize
        anchors = self._select_anchors(command)  # (B, M, T, 2)
        anchors_norm = self._normalize(anchors)

        # Add small noise to anchors (truncated starting point)
        noise = torch.randn_like(anchors_norm)
        init_timestep = self.t_trunc // self.num_inference_steps
        t_init = torch.full((bs * self.num_modes,), init_timestep, device=device, dtype=torch.long)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, device=device)

        noisy = self.scheduler.add_noise(
            anchors_norm.reshape(bs * self.num_modes, self.future_steps, 2),
            noise.reshape(bs * self.num_modes, self.future_steps, 2),
            t_init,
        )

        # Expand conditioning
        cond_flat = global_cond.unsqueeze(1).expand(-1, self.num_modes, -1).reshape(bs * self.num_modes, -1)

        # DDIM denoising steps
        step_ratio = self.t_trunc // self.num_inference_steps
        roll_timesteps = list(range(step_ratio * (self.num_inference_steps - 1), -1, -step_ratio))

        img = noisy
        for t_val in roll_timesteps:
            t_tensor = torch.full((bs * self.num_modes,), t_val, device=device, dtype=torch.long)
            pred_sample = self.unet(img, t_tensor, global_cond=cond_flat)
            pred_sample = torch.clamp(pred_sample, -1, 1)
            img = self.scheduler.step(
                model_output=pred_sample,
                timestep=t_val,
                sample=img,
            ).prev_sample

        # Denormalize and reshape
        result = self._denormalize(img.reshape(bs, self.num_modes, self.future_steps, 2))

        # Select best mode: use the one closest to its anchor (most confident)
        anchor_dist = ((result - self._denormalize(anchors_norm)) ** 2).sum(dim=-1).mean(dim=-1)  # (B, M)
        best_mode = anchor_dist.argmin(dim=-1)
        return result[torch.arange(bs, device=device), best_mode]

    def _normalize(self, traj: torch.Tensor) -> torch.Tensor:
        """Normalize trajectories to roughly [-1, 1] range."""
        # Simple scaling: divide by a fixed scale per axis
        # These will be set after anchor generation
        scale = self._get_scale(traj.device)
        return traj / scale

    def _denormalize(self, traj: torch.Tensor) -> torch.Tensor:
        """Inverse of _normalize."""
        scale = self._get_scale(traj.device)
        return traj * scale

    def _get_scale(self, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "_scale_buf"):
            # Default scales; will be overridden by set_normalization
            self.register_buffer("_scale_buf", torch.tensor([30.0, 5.0], device=device))
        return self._scale_buf

    def set_normalization(self, x_scale: float, y_scale: float):
        """Set normalization scale factors (from dataset statistics)."""
        self.register_buffer("_scale_buf", torch.tensor([x_scale, y_scale]))

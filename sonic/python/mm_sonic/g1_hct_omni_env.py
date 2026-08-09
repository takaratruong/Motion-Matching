"""Native G1 omnidirectional locomotion on broad challenging terrain.

This task is intentionally separate from the kinematic terrain-warp experiments.  It
warm-starts a dynamically valid IsaacLab G1 controller and broadens its command and
terrain distributions so its simulated states can be harvested as clean motion data.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import isaaclab.terrains as terrain_gen
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidSlopedTerrainCfg, HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HctVelocityCommand(UniformVelocityCommand):
    """Uniform two-stick commands with independently sampled inactive axes.

    Independent zeroing matters here: without it, a continuous uniform draw
    almost never exposes the controller to exact stops, pure strafes, pure
    spins, or straight-only walking.  The benchmark protocol deliberately
    zeroes command components independently for the same reason.
    """

    cfg: "HctVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._zero_component = torch.zeros(
            (self.num_envs, 3), dtype=torch.bool, device=self.device
        )

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        random = torch.rand((len(env_ids), 3), device=self.device)
        probability = torch.tensor(
            self.cfg.zero_component_probability,
            dtype=random.dtype,
            device=self.device,
        )
        self._zero_component[env_ids] = random < probability

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[self._zero_component] = 0.0


@configclass
class HctVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = HctVelocityCommand
    zero_component_probability: tuple[float, float, float] = (0.5, 0.5, 0.5)


@height_field_to_mesh
def rough_pyramid_slope_terrain(difficulty: float, cfg) -> np.ndarray:
    slope = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    rough = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg)
    return np.asarray(slope + rough, dtype=np.int16)


@configclass
class HfRoughPyramidSlopeTerrainCfg(HfPyramidSlopedTerrainCfg):
    """A smooth pyramid slope with low-amplitude correlated roughness."""

    function = rough_pyramid_slope_terrain
    noise_range: tuple[float, float] = (-0.025, 0.025)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


@height_field_to_mesh
def rolling_hills_terrain(difficulty: float, cfg) -> np.ndarray:
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    correlation = cfg.correlation_length_range[0] + difficulty * (
        cfg.correlation_length_range[1] - cfg.correlation_length_range[0]
    )

    noise = np.random.normal(size=(width, length))
    smooth = gaussian_filter(noise, sigma=max(correlation / cfg.horizontal_scale, 1.0), mode="reflect")
    smooth -= float(np.mean(smooth))
    scale = float(np.quantile(np.abs(smooth), 0.995))
    if scale > 1.0e-9:
        smooth *= amplitude / scale

    # Give reset poses a small, smoothly blended support patch without turning the
    # traversed scene into a large flat platform.
    if cfg.platform_width > 0.0:
        x = (np.arange(width) - 0.5 * (width - 1)) * cfg.horizontal_scale
        y = (np.arange(length) - 0.5 * (length - 1)) * cfg.horizontal_scale
        radius = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
        half = 0.5 * cfg.platform_width
        blend = np.clip((radius - half) / max(half, cfg.horizontal_scale), 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        center_height = float(smooth[width // 2, length // 2])
        smooth = center_height + blend * (smooth - center_height)

    return np.rint(smooth / cfg.vertical_scale).astype(np.int16)


@configclass
class HfRollingHillsTerrainCfg(HfTerrainBaseCfg):
    """Broad random hills rather than a repeated sinusoidal test surface."""

    function = rolling_hills_terrain
    amplitude_range: tuple[float, float] = MISSING
    correlation_length_range: tuple[float, float] = (0.55, 0.75)
    platform_width: float = 1.0


# The first six shares mirror the Humanoid Challenging Terrain families.  A
# smaller stair share is retained because the collected corpus must also cover
# ascending and descending stairs.  Scenes are wide 8 x 8 m tiles, not narrow
# approach-angle tricks.
HCT_OMNI_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1125,
            noise_range=(0.0, 0.005),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1125,
            noise_range=(-0.025, 0.025),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "smooth_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05625,
            slope_range=(0.02, 0.20),
            platform_width=1.5,
            border_width=0.25,
        ),
        "smooth_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05625,
            slope_range=(0.02, 0.20),
            platform_width=1.5,
            border_width=0.25,
        ),
        "rough_slope_up": HfRoughPyramidSlopeTerrainCfg(
            proportion=0.05625,
            slope_range=(0.02, 0.20),
            platform_width=1.5,
            border_width=0.25,
        ),
        "rough_slope_down": HfRoughPyramidSlopeTerrainCfg(
            proportion=0.05625,
            slope_range=(0.02, 0.20),
            platform_width=1.5,
            border_width=0.25,
            inverted=True,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.225,
            obstacle_width_range=(0.2, 0.8),
            obstacle_height_range=(0.005, 0.05),
            num_obstacles=70,
            platform_width=1.5,
            border_width=0.25,
        ),
        "rolling_hills": HfRollingHillsTerrainCfg(
            proportion=0.225,
            amplitude_range=(0.05, 0.30),
            correlation_length_range=(0.55, 0.75),
            platform_width=1.0,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.18),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


def terrain_relative_base_height_exp(
    env: "ManagerBasedRLEnv",
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Finite, bounded terrain-relative pelvis-height reward."""

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    finite = torch.isfinite(hits_z)
    count = finite.sum(dim=1)
    terrain_z = torch.where(finite, hits_z, torch.zeros_like(hits_z)).sum(dim=1) / count.clamp(min=1)
    error = asset.data.root_pos_w[:, 2] - (target_height + terrain_z)
    reward = torch.exp(-torch.square(error) / (std**2))
    reward = torch.where(count > 0, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


@configclass
class G1HctOmniEnvCfg(G1RoughEnvCfg):
    """Upright, quiet-arm G1 with body-relative two-stick commands."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = HCT_OMNI_TERRAINS_CFG
        # The stock velocity curriculum promotes terrain using net distance
        # from the spawn origin.  That is invalid for correct spins, reversals,
        # stops, arcs, and out-and-back joystick programs: all can finish near
        # the origin and be demoted.  Keep every difficulty represented instead
        # of allowing those commands to collapse training onto easy tiles.
        self.scene.terrain.max_init_terrain_level = HCT_OMNI_TERRAINS_CFG.num_rows - 1
        self.curriculum.terrain_levels = None

        self.commands.base_velocity = HctVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.5, 4.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            zero_component_probability=(0.5, 0.5, 0.5),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.8, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        )

        self.rewards.base_height = RewTerm(
            func=terrain_relative_base_height_exp,
            weight=1.0,
            params={
                "target_height": 0.74,
                "std": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        # Avoid a high-energy arm-swing signature in every collected clip.  This
        # stays soft so the arms remain available for balance on difficult ground.
        self.rewards.joint_deviation_arms.weight = -0.30


@configclass
class G1HctOmniRefinedEnvCfg(G1HctOmniEnvCfg):
    """Short refinement emphasizing faithful, planted joystick locomotion."""

    def __post_init__(self):
        super().__post_init__()
        # Phase-two visual review found robust traversal but under-response on
        # rightward commands and excessive stance-foot motion on the hardest
        # stairs.  These remain soft rewards: the exact contact and termination
        # mechanics are unchanged, and no terrain or command case is filtered.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.feet_slide.weight = -0.50
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.action_rate_l2.weight = -0.010


@configclass
class G1HctOmniPPORunnerCfg(G1RoughPPORunnerCfg):
    experiment_name = "g1_hct_omni"

    def __post_init__(self):
        super().__post_init__()
        # The selected upright seed was trained with RSL-RL's log-standard-
        # deviation parameterization.  Matching this explicitly keeps the
        # checkpoint load strict; a permissive load would silently change the
        # exploration distribution during warm-starting.
        self.policy.noise_std_type = "log"


gym.register(
    id="Terrain-G1-HCT-Omni-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "mm_sonic.g1_hct_omni_env:G1HctOmniEnvCfg",
        "rsl_rl_cfg_entry_point": "mm_sonic.g1_hct_omni_env:G1HctOmniPPORunnerCfg",
    },
)

gym.register(
    id="Terrain-G1-HCT-Omni-Refined-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "mm_sonic.g1_hct_omni_env:G1HctOmniRefinedEnvCfg",
        "rsl_rl_cfg_entry_point": "mm_sonic.g1_hct_omni_env:G1HctOmniPPORunnerCfg",
    },
)

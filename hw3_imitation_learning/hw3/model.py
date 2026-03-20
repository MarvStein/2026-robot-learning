"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        raise NotImplementedError


# Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int,
        depth: int
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.layers = nn.ModuleList()
        for i in range(depth):
            in_dim = state_dim if i == 0 else d_model
            out_dim = chunk_size * action_dim if i == depth - 1 else d_model
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                self.layers.append(nn.ReLU())

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = state
        for layer in self.layers:
            x = layer(x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state)

class ResidualMLP(nn.Module):
    """MLP with Residual Blocks and Dropout (used by MultiTaskPolicy)."""
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        chunk_size: int,
        d_model: int, 
        depth: int, 
        dropout: float = 0.1,
        skip_connections: bool = True
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.skip_connections = skip_connections
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(depth)
        ])
        
        # Final projection
        self.output_projection = nn.Linear(d_model, chunk_size * action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            if self.skip_connections:
                x = x + block(x)
            else:
                x = block(x)
        return self.output_projection(x).view(-1, self.chunk_size, self.action_dim)

# Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene.
    
    IMPORTANT
    ---------
    use `--state-keys state_ee_xyz state_gripper "original_pos_cube_red[:3]" "original_pos_cube_green[:3]" "original_pos_cube_blue[:3]" state_goal goal_pos`
    IN THIS EXACT ORDER for training.
    
    Implementation
    --------------
    Uses one-hot state_goal as hard attention weights over the three cubes, isolating
    the target cube and treating the task like a single-cube problem through an MLP.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int,
        depth: int,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # state[:4] is state_ee_xyz and state_gripper
        # state[4:7] is the xyz of red cube
        # state[7:10] is the xyz of green cube
        # state[10:13] is the xyz of blue cube
        # state[13:16] is state_goal
        # state[16:19] is goal_pos

        self.mlp_input_dim = 7 # gripper, ee_to_cube (3), ee_to_goal (3)
        
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)

        # Reuse ObstaclePolicy as MLP (no obstacle)
        self.mlp = ResidualMLP(
            state_dim=self.mlp_input_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth
        )

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Intercept load_state_dict to dynamically add buffers if they are missing."""
        if "state_mean" in state_dict and getattr(self, "state_mean", None) is None:
            self.register_buffer("state_mean", state_dict["state_mean"])
        if "state_std" in state_dict and getattr(self, "state_std", None) is None:
            self.register_buffer("state_std", state_dict["state_std"])
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        B = state.shape[0]
        
        gripper = state[:, 3:4] # (B, 1)
        # we undo the normalization for two reasons:
        # 1.    state_goal should be truly one-hot (1.00 - mean)/std != 1.00
        # 2.    The MLP should be indifferent to which cube is actually the target ("attended_cube")
        #       and thus the cube position should always use the same scaling.
        #       i.e. we undo the normalization and then apply (WLOG) the normalization of the red cube to the attended_cube
        unnormalized_state = (state * self.state_std) + self.state_mean
        # sanity check to be 100% sure we have binary values (1.00 and 0.00) even if
        # undoing the normalization leads to small numerical inaccuracies.
        state_goal = nn.functional.one_hot(unnormalized_state[:, 13:16].argmax(dim=1), num_classes=3).float() # (B, 3)
        robot_xyz = unnormalized_state[:, :3] # (B, 3) # use unnormalized such that substracting attended_cube from robot_xyz is in the same scale
        goal_pos = unnormalized_state[:, 16:19]
        
        # Attention with one-hot state_goal as weights to isolate target cube
        weights = state_goal.unsqueeze(-1)
        cubes = unnormalized_state[:, 4:13].view(B, 3, 3) # (B, 3, 3), second dim is cube index, third dim is xyz of cube
        attended_cube = (cubes * weights).sum(dim=1)
        # e.g. cubes = [[[x_r, y_r, z_r], [x_g, y_g, z_g], [x_b, y_b, z_b]]]
        # state_goal = [[0, 1, 0]] (green)
        # weights = [[[0], [1], [0]]]
        # cubes * weights = [[[0, 0, 0],[x_g, y_g, z_g], [0, 0, 0]]]
        # attended_cube = [[x_g, y_g, z_g]]

        # normalize attended_cube but use same mean and std no matter which cube it is
        # we use mean and std of the red cube but (hopefully) doesn't matter which one we used.
        # attended_cube = (attended_cube - self.state_mean[4:7]) / self.state_std[4:7]
        ee_to_cube = attended_cube - robot_xyz # (B, 3)
        ee_to_goal = goal_pos - robot_xyz # (B, 3)
        
        x = torch.cat([gripper, ee_to_cube, ee_to_goal], dim=1)
        return self.mlp(x)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int,
    depth: int,
    state_mean: torch.Tensor | None = None,
    state_std: torch.Tensor | None = None
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
        )
    if policy_type == "multitask":
        # IMPORTANT:
        # for MultiTaskPolicy, state_mean and state_std should not be left None but
        # should be set to the actual values from the normalizer
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth,
            state_mean=state_mean,
            state_std=state_std
        )
    raise ValueError(f"Unknown policy type: {policy_type}")

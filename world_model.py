import torch
import torch.nn as nn
from task_embed import TaskEmbed
from torch.distributions import Categorical


class MiniWorldModel(nn.Module):
    def __init__(self, task_embed: TaskEmbed, num_actions=18):
        super().__init__()
        self.num_actions = num_actions
        self.action_embed = nn.Embedding(
            num_actions, 256
        )  # num_actions tokens, 256 dim
        self.task_embed = task_embed

        self.pos = nn.Parameter(
            torch.randn(1, 1024, 256)
        )  # positional encoding for 64 tokens

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=1024,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.obs_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 16 * 512),  # output obs embedding
        )

        self.reward_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def rollout(self, initial_obs_tokens, horizon, actor_critic, game_id):
        """
        Simulate a rollout from initial observation tokens
        initial_obs_tokens: (B, 16)
        horizon: int
        actor_critic: ActorCritic model
        game_id: (B,)
        returns: states (list of (B, 16)), actions (list of (B,)), rewards (list of (B,))
        """
        states = [initial_obs_tokens.unsqueeze(1)]  # List of (B, 1, 16)
        actions, rewards, values, value_vars = [], [], [], []

        for t in range(horizon):
            # build full history for policy
            seq_tokens = torch.cat(states, dim=1)  # (B, t+1, 16)

            # forward pass through actor-critic
            action_logits, value, value_var = actor_critic(
                seq_tokens
            )  # (B, t+1, num_actions), (B, t+1, 1), (B, t+1, 1)
            action = Categorical(
                logits=action_logits[:, -1, :]
            ).sample()  # (B,)

            actions.append(action.unsqueeze(1))  # (B, 1)
            values.append(value[:, -1, :])  # (B, 1)
            value_vars.append(value_var[:, -1, :])  # (B, 1)

            with torch.no_grad():
                full_obs = seq_tokens.detach()  # (B, t+1, 16)
                full_actions = torch.cat(actions, dim=1)  # (B, t+1)

                pred_obs_logits, pred_reward = self(
                    full_obs, full_actions, game_id
                )  # (B, t+1, 16, 512), (B, t+1)

                next_tokens = Categorical(
                    logits=pred_obs_logits[:, -1, :, :]
                ).sample()  # (B, 16)
                pred_reward = pred_reward[:, -1:]  # (B, 1)

            states.append(next_tokens.unsqueeze(1))  # (B, t+2, 16)
            rewards.append(pred_reward)  # (B, 1)

        states = torch.cat(states[1:], dim=1)  # (B, horizon, 16)
        actions = torch.cat(actions, dim=1)  # (B, horizon)
        rewards = torch.cat(rewards, dim=1)  # (B, horizon)
        values = torch.cat(values, dim=1)  # (B, horizon)
        value_vars = torch.cat(value_vars, dim=1)  # (B, horizon)

        return states, actions, rewards, values, value_vars

    def forward(self, obs_tokens, action_tokens, game_ids):
        """
        obs_tokens: (B, T, K)  ← each obs is K tokens
        action_tokens: (B, T)  ← each timestep has 1 discrete action
        game_ids: (B,)  ← each timestep has 1 discrete game id
        Returns:
            pred_obs_logits: (B, T, K, 512)
            pred_rewards:    (B, T)
        """
        assert (
            action_tokens.max().item() < self.num_actions
        ), f"Action token {action_tokens.max().item()} exceeds embedding limit {self.num_actions - 1}"

        B, T, K = obs_tokens.shape

        # embed tokens
        act_embed = self.action_embed(action_tokens).unsqueeze(
            2
        )  # (B, T, 1, 256)

        obs_embed_with_task = self.task_embed(
            game_ids=game_ids, obs_tokens=obs_tokens
        )  # (B, T*K, 256)

        # interleave
        combined_embed = torch.cat(
            [obs_embed_with_task.reshape(B, T, K, -1), act_embed], dim=2
        ).reshape(
            B, T * (K + 1), 256
        )  # (B, T, K+1, 256) -> (B, T*(K+1), 256)

        # positional encoding
        pos = self.pos[:, : combined_embed.size(1), :]  # (1, T*(K+1), 256)

        # add positional encoding and task embedding
        x = combined_embed + pos  # (B, T*(K+1), 256)

        # Causal mask to prevent looking ahead
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )  # (T*(K+1), T*(K+1))

        # decode
        out = self.transformer(x, mask, is_causal=True)  # (B, T*(K+1), 256)
        out = out.reshape(B, T, K + 1, 256)  # (B, T, K+1, 256)

        action_features = out[:, :, -1, :]  # (B, T, 256)

        obs_logits = self.obs_head(action_features).reshape(
            B, T, K, 512
        )  # (B, T, K, 512)
        rewards = self.reward_head(action_features).squeeze(-1)  # (B, T)

        return obs_logits, rewards

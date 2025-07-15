import torch
import torch.nn as nn
import torch.nn.functional as F
from task_embed import TaskEmbed
from wandb import init


class ActorCritic(nn.Module):
    def __init__(self, task_embed: TaskEmbed, num_actions=18, max_seq_len=32):
        super().__init__()
        self.cond_proj = nn.Linear(256, 256)
        self.task_embed = task_embed
        self.pos = nn.Parameter(torch.randn(1, max_seq_len * 16, 256))

        # Learnable query for attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, 256))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=4,
                activation="gelu",
                dim_feedforward=1024,
                batch_first=True,
                norm_first=True,
                dropout=0.1,
            ),
            num_layers=6,
        )

        self.actor_temp = nn.Parameter(torch.tensor(1.0))
        self.actor_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(512, num_actions),  # num_actions
        )

        # Critic head with uncertainty (2 outputs: mean and log-variance)
        self.critic_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(512, 2),  # [mean, log_variance] for Gaussian uncertainty
        )

    def forward(self, obs_tokens, game_ids=None):
        """
        obs_tokens: token ids (B, T, K)
        returns:
        - action_logits: (B, T, num_actions)
        - values: (B, T, 1) mean value
        - value_vars: (B, T, 1) uncertainty variance
        """
        B, T, K = obs_tokens.shape

        obs_embed_with_task = self.task_embed(
            game_ids=game_ids, obs_tokens=obs_tokens
        )  # (B, T*K, 256)

        # positional encoding
        pos = self.pos[:, : obs_embed_with_task.size(1), :]  # (1, T*K, 256)

        # add positional encoding and task embedding
        x = obs_embed_with_task + pos  # (B, T*K, 256)

        # Causal mask to prevent looking ahead
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )  # (T*K, T*K)

        out = self.transformer(x, mask, is_causal=True)  # (B, T*K, 256)
        out = out.reshape(B, T, K, 256)  # (B, T, 16, 256)

        # Attention pooling
        query = self.pool_query.repeat(B, T, 1)  # (B, T, 256)
        attn_weights = F.softmax(
            query @ out.transpose(-2, -1) / (256**0.5), dim=-1
        )  # (B, T, 1, 16)
        out = (attn_weights @ out).squeeze(2)  # (B, T, 256)

        # Dynamic conditioning
        obs_mean = out.mean(dim=-1, keepdim=True)  # (B, T, 1)
        cond = F.tanh(self.cond_proj(obs_mean.squeeze(-1)))  # (B, T, 256)
        out = out + 0.1 * cond  # (B, T, 256)

        action_logits = self.action_head(out) / self.actor_temp.clamp(min=0.1)

        critic_out = self.critic_head(out)  # (B, T, 2)
        values = critic_out[:, :, 0:1]  # Mean value (B, T, 1)
        log_vars = critic_out[:, :, 1:2]  # Log-variance (B, T, 1)
        value_vars = torch.exp(log_vars)  # Variance (B, T, 1)

        return action_logits, values, value_vars

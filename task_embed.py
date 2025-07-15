import torch
import torch.nn as nn


class TaskEmbed(nn.Module):
    def __init__(
        self,
        num_known_tasks=1,
        token_vocab=512,
        embed_dim=256,
        feature_dim=256,
    ):
        super().__init__()
        self.known_embed = nn.Embedding(
            num_known_tasks + 5, embed_dim
        )  # +10 for buffer new games
        self.token_embed = nn.Embedding(token_vocab, feature_dim)
        self.infer_mlp = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blend_ratio = nn.Parameter(torch.tensor(0.5))
        self.is_known = True

    def forward(
        self,
        obs_tokens,
        game_ids=None,
    ):
        """
        game_ids: (B,)  ← each game id corresponds to a known task
        obs_tokens: (B, T, K)  ← each obs is K tokens
        Returns:
            obs_embed_with_task: (B, T*K, feature_dim)
        """
        B, T, K = obs_tokens.shape

        obs_tokens_flat = obs_tokens.reshape(B, T * K)  # (B, T*K)
        obs_embed = self.token_embed(obs_tokens_flat)  # (B, T*K, feature_dim)

        obs_embed_flat = obs_embed.mean(dim=1)  # (B, feature_dim)
        infer_task_embed = self.infer_mlp(obs_embed_flat)  # (B, embed_dim)

        if self.is_known and game_ids is not None:
            known_task_embed = self.known_embed(game_ids)  # (B, embed_dim)
            task_embed = (
                known_task_embed * (1 - self.blend_ratio) + infer_task_embed
            ) * self.blend_ratio
        else:
            task_embed = infer_task_embed

        obs_embed_with_task = obs_embed + task_embed.unsqueeze(1).repeat(
            1, T * K, 1
        )  # (B, T*K, feature_dim)

        return obs_embed_with_task

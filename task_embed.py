import torch
import torch.nn as nn


class TaskEmbed(nn.Module):
    def __init__(
        self,
        num_known_tasks=6,
        token_vocab=512,
        action_vocab=18,
        embed_dim=256,
        feature_dim=256,
    ):
        super().__init__()
        self.known_embedding = nn.Embedding(
            num_known_tasks * 2, embed_dim
        )  # +10 for buffer new games
        self.infer_mlp = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.blend_ratio = nn.Parameter(torch.tensor(0.5))

        self.obs_embedding = nn.Embedding(token_vocab, feature_dim)
        self.action_embedding = nn.Embedding(action_vocab + 1, feature_dim)

        self.pool_query1 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pool_query2 = nn.Parameter(torch.randn(1, 1, feature_dim))

        # FiLM layers for task conditioning
        self.obs_scale = nn.Linear(embed_dim, feature_dim)
        self.obs_bias = nn.Linear(embed_dim, feature_dim)
        self.act_scale = nn.Linear(embed_dim, feature_dim)
        self.act_bias = nn.Linear(embed_dim, feature_dim)

        nn.init.zeros_(self.obs_scale.weight)
        nn.init.ones_(self.obs_scale.bias)
        nn.init.zeros_(self.obs_bias.weight)
        nn.init.zeros_(self.obs_bias.bias)
        nn.init.zeros_(self.act_scale.weight)
        nn.init.ones_(self.act_scale.bias)
        nn.init.zeros_(self.act_bias.weight)
        nn.init.zeros_(self.act_bias.bias)

    def forward(
        self, obs_tokens, action_tokens=None, game_ids=None, is_known=True
    ):
        """
        obs_tokens: (B, T, K)  ← each obs is K tokens
        action_tokens: (B, T)  ← each timestep has 1 discrete action
        game_ids: (B,)  ← each game id corresponds to a known task
        Returns:
            obs_embed_with_task: (B, T*K, embed_dim)
            action_embed_with_task: (B, T, embed_dim) if action_tokens is not
            task_embed: (B, embed_dim)
        """
        B, T, K = obs_tokens.shape

        obs_embed = self.obs_embedding(obs_tokens)  # (B, T, K, feature_dim)
        action_embed = (
            self.action_embedding(action_tokens)
            if action_tokens is not None
            else None
        )  # (B, T, feature_dim)

        # attention pooling over tokens
        query = self.pool_query1.repeat(B, T, 1).unsqueeze(
            2
        )  # (B, T, 1, feature_dim)
        attn_weights = torch.softmax(
            query  # (B, T, 1, feature_dim)
            @ obs_embed.transpose(-2, -1)  # (B, T, feature_dim, K)
            / (256**0.5),
            dim=-1,
        )  # (B, T, 1, K)
        pooled = (attn_weights @ obs_embed).squeeze(2)  # (B, T, feature_dim)

        # attention pooling over time steps
        query = self.pool_query2.repeat(B, 1, 1)  # (B, 1, feature_dim)
        attn_weights = torch.softmax(
            query  # (B, 1, feature_dim)
            @ pooled.transpose(-2, -1)  # (B, T, feature_dim)
            / (256**0.5),
            dim=-1,
        )
        pooled = (attn_weights @ pooled).squeeze(1)  # (B, feature_dim)

        # infer task embedding from pooled features
        infer_task_embed = self.infer_mlp(pooled)  # (B, embed_dim)

        if is_known and game_ids is not None:
            known_task_embed = self.known_embedding(game_ids)  # (B, embed_dim)
            r = torch.sigmoid(self.blend_ratio)
            task_embed = known_task_embed * (1 - r) + infer_task_embed * r
        else:
            task_embed = infer_task_embed

        obs_scale = self.obs_scale(task_embed)  # (B, feature_dim)
        obs_bias = self.obs_bias(task_embed)  # (B, feature_dim)
        act_scale = self.act_scale(task_embed)  # (B, feature_dim)
        act_bias = self.act_bias(task_embed)  # (B, feature_dim)

        obs_embed = obs_embed.reshape(B, T * K, -1)  # (B, T*K, feature_dim)
        obs_embed_with_task = obs_embed * obs_scale.unsqueeze(
            1
        ) + obs_bias.unsqueeze(
            1
        )  # (B, T*K, embed_dim)

        if action_embed is not None:
            action_embed_with_task = action_embed * act_scale.unsqueeze(
                1
            ) + act_bias.unsqueeze(
                1
            )  # (B, T, embed_dim)

        return (
            obs_embed_with_task,  # (B, T*K, embed_dim)
            (
                action_embed_with_task if action_embed is not None else None
            ),  # (B, T, embed_dim)
        )

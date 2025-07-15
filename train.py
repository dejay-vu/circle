# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from einops import rearrange
from safetensors.torch import load_file
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    Subset,
)
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm, trange
from vector_quantize_pytorch import VectorQuantize

encoder_state = load_file("pretrained/encoder.safetensors")
vq_state = load_file("pretrained/vq.safetensors")
decoder_state = load_file("pretrained/decoder.safetensors")


# %%
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.net(x))


class Encoder(nn.Module):
    """
    (B, 3, 84, 84) → (B, 16, 512)

    - 4 conv layers
    - 2 residual blocks per layer
    - 21x21 feature map partitioned into 4x4 non-overlapping 5x5 windows -> 16 tokens
    """

    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        self.patch = 5

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(64),
            ResidualBlock(64),
        )  # 84 → 42

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )  # 42 → 21

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            ResidualBlock(256),
            ResidualBlock(256),
        )  # 21 → 21

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )  # 21 → 21

        self.projection = nn.Linear(
            self.patch * self.patch * out_channels, out_channels
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # (B, 512, 21, 21)
        x = x[:, :, 0:20, 0:20]
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch,
            p2=self.patch,
        )  # (B, 16, 5*5*512)

        x = self.projection(x)  # (B, 16, 512)

        return x  # (B, 16, 512)


class Decoder(nn.Module):
    """
    (B, 16, 512) → (B, 3, 84, 84)

    - reverse of Encoder
    - 16 tokens -> 4x4 grid of 5x5 patches -> (B, 512, 21, 21)
    - 4 deconv layers
    """

    def __init__(self, in_channels=512, out_channels=3):
        super().__init__()
        self.patch = 5

        self.projection = nn.Linear(
            in_channels, self.patch * self.patch * in_channels
        )

        self.deconv4 = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
        )  # 21 → 21

        self.deconv3 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.SiLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        )  # 21 → 21

        self.deconv2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        )  # 21 → 42

        self.deconv1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.SiLU(),
            nn.ConvTranspose2d(
                64, out_channels, kernel_size=4, stride=2, padding=1
            ),
        )  # 42 → 84

    def forward(self, x):
        # x: (B, 16, 512)
        x = self.projection(x)  # (B, 16, 5×5×512)
        x = x.view(
            x.size(0), 4, 4, 512, self.patch, self.patch
        )  # (B, 4, 4, 512, 5, 5)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, 512, 4, 5, 4, 5)
        x = x.reshape(x.size(0), 512, 4 * 5, 4 * 5)  # (B, 512, 20, 20)

        # 填充到 21×21（补 1 行 1 列）
        x = F.pad(x, (0, 1, 0, 1))  # (B, 512, 21, 21)

        x = self.deconv4(x)  # (B, 256, 21, 21)
        x = self.deconv3(x)  # (B, 128, 21, 21)
        x = self.deconv2(x)  # (B, 64, 42, 42)
        x = self.deconv1(x)  # (B, 3, 84, 84)

        return x


# %%
class MiniWorldModel(nn.Module):
    def __init__(self, num_actions=18, num_games=6):
        super().__init__()
        self.num_actions = num_actions
        self.obs_embed = nn.Embedding(512, 256)  # 512 tokens, 256 dim
        self.action_embed = nn.Embedding(
            num_actions, 256
        )  # num_actions tokens, 256 dim
        self.game_embed = nn.Embedding(
            num_games, 256
        )  # num_games tokens, 256 dim

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
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.obs_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),  # output obs embedding
        )

        self.reward_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def rollout(self, obs_tokens, action_tokens, actor_critic):
        """
        obs_tokens: token ids from VQ-VAE -> (B, 16)
        action_tokens: token ids for actions -> (B,)
        actor_critic: ActorCritic model to compute next state and reward
        returns: next_obs_logits, reward
        """
        pass

    def forward(self, obs_tokens, action_tokens, game_ids):
        """
        obs_tokens: (B, T, K)  ← each obs is K tokens
        action_tokens: (B, T)  ← each timestep has 1 discrete action
        game_ids: (B, T)  ← each timestep has 1 discrete game id
        Returns:
            pred_obs_logits: (B, T, K, vocab_size)
            pred_rewards:    (B, T)
        """
        assert (
            action_tokens.max().item() < self.num_actions
        ), f"Action token {action_tokens.max().item()} exceeds embedding limit {self.num_actions - 1}"

        B, T, K = obs_tokens.shape

        # flatten for embedding
        z = obs_tokens.reshape(B, T * K)  # (B, T*K)
        a = action_tokens  # (B, T)

        # embed tokens
        z_embed = self.obs_embed(z)  # (B, T*K, 256)
        a_embed = self.action_embed(a).unsqueeze(2)  # (B, T, 1, 256)

        g_embed = (
            self.game_embed(game_ids).unsqueeze(1).repeat(1, T * (K + 1), 1)
        )  # (B, T*(K+1), 256)

        # interleave
        tokens = torch.cat(
            [z_embed.view(B, T, K, -1), a_embed], dim=2
        )  # (B, T, K+1, 256)
        tokens = tokens.reshape(B, T * (K + 1), 256)  # (B, T*(K+1), 256)

        # add positional encoding
        pos_embed = self.pos[:, : tokens.size(1), :]  # (1, T*(K+1), 256)
        x = tokens + pos_embed + g_embed  # (B, T*(K+1), 256)

        # causal mask (L, L)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )  # (T*(K+1), T*(K+1))

        # decode
        out = self.transformer(x, mask, is_causal=True)  # (B, T*(K+1), 256)

        out = out.reshape(B, T, K + 1, 256)  # (B, T, K+1, 256)

        obs_logits = self.obs_head(out[:, :, :-1, :])  # (B, T, K, 512)
        rewards = self.reward_head(out[:, :, -1, :]).squeeze(-1)  # (B, T)

        return obs_logits, rewards


# %%
class ActorCritic(nn.Module):
    def __init__(self, num_actions=18):
        super().__init__()
        self.embed = nn.Embedding(512, 512)
        self.pos = nn.Parameter(torch.randn(1, 16, 512))

        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                activation="gelu",
                dim_feedforward=1024,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=6,
        )

        self.actor_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_actions),  # num_actions
        )

        self.critic_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        """
        x: token ids from VQ-VAE -> (B, 16)
        returns:
        - action logits: (B, 18)
        - value: (B, 1)
        """
        x = self.embed(x) + self.pos[:, : x.size(1), :]
        x = self.blocks(x)
        x = x.mean(dim=1)  # (B, 512)

        action_logits = self.actor_head(x)
        value = self.critic_head(x)

        return action_logits, value


# %%
class MultiTrajectoryDataset(Dataset):
    def __init__(self, games, horizon=8):
        self.root = zarr.open_group("dataset50k.zarr", mode="r")
        self.games = games
        self.horizon = horizon
        self.data = []

        for gid, game in enumerate(self.games):
            frames = self.root[game]["frames"][:]
            actions = self.root[game]["actions"][:]
            rewards = self.root[game]["rewards"][:]
            dones = self.root[game]["dones"][:]

            for idx in trange(len(frames) - horizon):
                # check if any done in the horizon (to avoid crossing episode boundary)
                if dones[idx : idx + horizon].any():
                    continue

                frame_seq = frames[idx : idx + horizon, -1]  # (H, 3, 84, 84)
                action_seq = actions[idx : idx + horizon]  # (H,)
                reward_seq = rewards[idx : idx + horizon]
                reward_seq = (reward_seq - reward_seq.mean()) / (
                    reward_seq.std() + 1e-6
                )
                self.data.append((frame_seq, action_seq, reward_seq, gid))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_seq, action_seq, reward_seq, gid = self.data[idx]

        return (
            torch.from_numpy(frame_seq).float().div_(255),  # (H, 3, 84, 84)
            torch.from_numpy(action_seq).long(),  # (H,)
            torch.from_numpy(reward_seq).float(),  # (H,)
            torch.tensor(gid, dtype=torch.long),  # game id
        )


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
world_model = MiniWorldModel(num_actions=18, num_games=6).to(device)

# Load pretrained weights
encoder = Encoder().to(device)
quantizer = VectorQuantize(
    dim=512,
    codebook_size=512,  # each table smaller
    decay=0.8,
    commitment_weight=0.1,
).to(device)
encoder.load_state_dict(encoder_state)
quantizer.load_state_dict(vq_state)

world_model.train()
encoder.eval()
quantizer.eval()

optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)

# %%
games = [
    "SpaceInvaders",
    "Krull",
    "BeamRider",
    "Hero",
    "StarGunner",
    "MsPacman",
]
dataset = MultiTrajectoryDataset(games, horizon=8)

rng = np.random.default_rng(seed=42)
indices = np.arange(len(dataset))
perm = rng.permutation(indices)
split = int(len(perm) * 0.9)
train_indices = perm[:split]
val_indices = perm[split:]

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=256,
    sampler=RandomSampler(train_subset, replacement=True),
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_subset,
    batch_size=256,
    sampler=SequentialSampler(val_subset),
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
)

# %%
import os

import wandb

run = wandb.init(
    project="pretrain-world-model",
    name="Multigames-50k",
    config={
        "batch_size": 256,
        "epochs": 100,
        "learning_rate": 1e-4,
        "horizon": 8,
    },
)

best_val_loss = float("inf")
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Early stopping参数
patience = 5  # 连续5次val不改善就stop
early_stop_counter = 0
min_delta = 0.0001  # 最小改善阈值

global_step = 0
for epoch in range(100):
    train_game_obs_loss = {gid: 0.0 for gid in range(6)}
    train_game_reward_loss = {gid: 0.0 for gid in range(6)}
    train_game_count = {gid: 0 for gid in range(6)}

    bar = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1:02d}")
    world_model.train()
    for frames, actions, rewards, game_ids in bar:
        global_step += 1

        B, H, C, Ht, Wt = frames.shape

        frames = frames.view(-1, 3, 84, 84).to(
            device, dtype=torch.float32, non_blocking=True
        )
        actions = actions.to(device, dtype=torch.long, non_blocking=True)
        rewards = rewards.to(device, dtype=torch.float32, non_blocking=True)
        game_ids = game_ids.to(device)

        with torch.no_grad():
            z_e = encoder(frames)
            _, indices, _ = quantizer(z_e)
            obs_tokens = indices.view(B, H, 16)

        pred_obs_logits, pred_rewards = world_model(
            obs_tokens, actions, game_ids
        )

        pred_obs_logits = pred_obs_logits.permute(0, 1, 3, 2)

        obs_loss = F.cross_entropy(
            pred_obs_logits.reshape(-1, 512),
            obs_tokens.reshape(-1),
            reduction="mean",
        )

        reward_target = (rewards.abs() > 1e-6).float()
        reward_loss = sigmoid_focal_loss(
            pred_rewards,
            reward_target,
            reduction="mean",
            alpha=0.25,
            gamma=2.0,
        )

        loss = obs_loss + reward_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(B):
            gid = game_ids[i].item()
            train_game_obs_loss[gid] += obs_loss.item() / B
            train_game_reward_loss[gid] += reward_loss.item() / B
            train_game_count[gid] += 1

        bar.set_postfix(
            obs_loss=obs_loss.item(),
            reward_loss=reward_loss.item(),
            loss=loss.item(),
        )

        if global_step % 500 == 0:
            obs_loss_sum = 0.0
            reward_loss_sum = 0.0
            token_cnt = 0
            tp = pos = fp = 0  # 初始化整体

            val_game_obs_loss_sum = {gid: 0.0 for gid in range(6)}
            val_game_reward_loss_sum = {gid: 0.0 for gid in range(6)}
            val_game_token_cnt = {gid: 0 for gid in range(6)}
            val_game_tp = {gid: 0 for gid in range(6)}
            val_game_pos = {gid: 0 for gid in range(6)}
            val_game_fp = {gid: 0 for gid in range(6)}

            val_bar = tqdm(
                val_loader,
                leave=False,
                desc=f"Validating at step {global_step}",
            )
            world_model.eval()
            with torch.no_grad():
                for frames, actions, rewards, game_ids in val_bar:
                    B, H, C, Ht, Wt = frames.shape

                    frames = frames.view(-1, 3, 84, 84).to(
                        device, dtype=torch.float32, non_blocking=True
                    )
                    actions = actions.to(
                        device, dtype=torch.long, non_blocking=True
                    )
                    rewards = rewards.to(
                        device, dtype=torch.float32, non_blocking=True
                    )
                    game_ids = game_ids.to(device)

                    z_e = encoder(frames)
                    _, indices, _ = quantizer(z_e)
                    obs_tokens = indices.view(B, H, 16)

                    pred_obs_logits, pred_rewards = world_model(
                        obs_tokens, actions, game_ids
                    )
                    pred_obs_logits = pred_obs_logits.permute(0, 1, 3, 2)

                    obs_loss = F.cross_entropy(
                        pred_obs_logits.reshape(-1, 512),
                        obs_tokens.reshape(-1),
                        reduction="mean",
                    )
                    reward_target = (rewards.abs() > 1e-6).float()
                    reward_loss = sigmoid_focal_loss(
                        pred_rewards,
                        reward_target,
                        reduction="mean",
                        alpha=0.25,
                        gamma=2.0,
                    )

                    batch_tokens = obs_tokens.numel()
                    obs_loss_sum += obs_loss.item() * batch_tokens
                    reward_loss_sum += reward_loss.item() * batch_tokens
                    token_cnt += batch_tokens

                    batch_tokens_per_sample = obs_tokens.numel() // B
                    for i in range(B):
                        gid = game_ids[i].item()
                        val_game_obs_loss_sum[gid] += (
                            obs_loss.item() * batch_tokens_per_sample
                        )
                        val_game_reward_loss_sum[gid] += (
                            reward_loss.item() * batch_tokens_per_sample
                        )
                        val_game_token_cnt[gid] += batch_tokens_per_sample

                        prob_i = torch.sigmoid(pred_rewards[i])
                        pred_i = prob_i > 0.5
                        target_i = reward_target[i]
                        val_game_tp[gid] += (
                            (pred_i & target_i.bool()).sum().item()
                        )
                        val_game_pos[gid] += target_i.sum().item()
                        val_game_fp[gid] += (
                            (pred_i & (~target_i.bool())).sum().item()
                        )

                    # 取消注释：如果需要整体tp等
                    prob = torch.sigmoid(pred_rewards)
                    pred = prob > 0.5
                    tp += (pred & reward_target.bool()).sum().item()
                    pos += reward_target.sum().item()
                    fp += (pred & (~reward_target.bool())).sum().item()

            for gid in range(6):
                if val_game_token_cnt[gid] > 0:
                    obs_loss_game = (
                        val_game_obs_loss_sum[gid] / val_game_token_cnt[gid]
                    )
                    reward_loss_game = (
                        val_game_reward_loss_sum[gid] / val_game_token_cnt[gid]
                    )
                    total_loss_game = obs_loss_game + reward_loss_game
                    recall_game = (
                        val_game_tp[gid] / val_game_pos[gid]
                        if val_game_pos[gid]
                        else float("nan")
                    )
                    precision_game = (
                        val_game_tp[gid]
                        / (val_game_tp[gid] + val_game_fp[gid])
                        if (val_game_tp[gid] + val_game_fp[gid])
                        else float("nan")
                    )
                    f1_game = (
                        (
                            2
                            * precision_game
                            * recall_game
                            / (precision_game + recall_game)
                        )
                        if precision_game and recall_game
                        else float("nan")
                    )

                    wandb.log(
                        {
                            f"val/game_{games[gid]}/obs_loss": obs_loss_game,
                            f"val/game_{games[gid]}/reward_loss": reward_loss_game,
                            f"val/game_{games[gid]}/total_loss": total_loss_game,
                            f"val/game_{games[gid]}/recall": recall_game,
                            f"val/game_{games[gid]}/precision": precision_game,
                            f"val/game_{games[gid]}/f1": f1_game,
                        },
                        step=global_step,
                    )

            obs_loss_epoch = obs_loss_sum / token_cnt
            reward_loss_epoch = reward_loss_sum / token_cnt
            total_loss_epoch = obs_loss_epoch + reward_loss_epoch

            recall = tp / pos if pos else float("nan")
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall
                else float("nan")
            )

            wandb.log(
                {
                    "all_val/obs_loss": obs_loss_epoch,
                    "all_val/reward_loss": reward_loss_epoch,
                    "all_val/total_loss": total_loss_epoch,
                    "all_val/recall": recall,
                    "all_val/precision": precision,
                    "all_val/f1": f1,
                },
                step=global_step,
            )

            if total_loss_epoch < best_val_loss - min_delta:
                best_val_loss = total_loss_epoch
                early_stop_counter = 0
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"best_model_step_{global_step}.pth"
                )
                torch.save(world_model.state_dict(), checkpoint_path)
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at step {global_step}")
                    break

    if early_stop_counter >= patience:
        break

    for gid in range(6):
        if train_game_count[gid] > 0:
            wandb.log(
                {
                    f"train/game_{games[gid]}/obs_loss": train_game_obs_loss[
                        gid
                    ]
                    / train_game_count[gid],
                    f"train/game_{games[gid]}/reward_loss": train_game_reward_loss[
                        gid
                    ]
                    / train_game_count[gid],
                },
                step=global_step,
            )

run.finish()

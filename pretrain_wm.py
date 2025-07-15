# %%
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import zarr
from safetensors.torch import load_file, save_file
from task_embed import TaskEmbed
from torch.amp import GradScaler, autocast
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    Subset,
)
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm, trange
from vq_vae import Encoder, vq_vae
from world_model import MiniWorldModel

is_amp = True

run = wandb.init(
    project="pretrain-world-model",
    name="Multigames-50k",
    config={
        "batch_size": 256,
        "epochs": 1000,
        "learning_rate": 1e-4,
        "horizon": 16,
    },
)

encoder_state = load_file("pretrained/encoder.safetensors")
vq_state = load_file("pretrained/vq.safetensors")
decoder_state = load_file("pretrained/decoder.safetensors")


# %%
class MultiTrajectoryDataset(Dataset):
    def __init__(self, games, horizon=8):
        self.root = zarr.open_group("dataset50k.zarr", mode="r")
        self.games = games
        self.horizon = horizon
        self.data = []
        self.weights = []

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
                reward_seq = rewards[idx : idx + horizon]  # (H,)

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
# Task embedding
task_embed = TaskEmbed(num_known_tasks=6).to(device)

# Initialize models
world_model = MiniWorldModel(num_actions=18, task_embed=task_embed).to(device)

# Load pretrained weights
encoder = Encoder().to(device)
quantizer = vq_vae.to(device)
encoder.load_state_dict(encoder_state)
quantizer.load_state_dict(vq_state)

world_model.train()
encoder.eval()
quantizer.eval()

optimizer = torch.optim.AdamW(
    [
        {
            "params": (
                list(world_model.transformer.parameters())
                + list(world_model.action_embed.parameters())
                + list(world_model.task_embed.parameters())
                + [world_model.pos]
            ),
            "lr": 3e-4,
            "weight_decay": 1e-5,
        },
        {
            "params": (list(world_model.obs_head.parameters())),
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        {
            "params": (list(world_model.reward_head.parameters())),
            "lr": 1e-5,
            "weight_decay": 0.0,
        },
    ]
)

# scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# %%
games = [
    "SpaceInvaders",
    "Krull",
    "BeamRider",
    "Hero",
    "StarGunner",
    "MsPacman",
]
dataset = MultiTrajectoryDataset(games, horizon=10)

rng = np.random.default_rng(seed=42)
indices = np.arange(len(dataset))
perm = rng.permutation(indices)
split = int(len(perm) * 0.9)
train_indices = perm[:split]
val_indices = perm[split:]

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

train_sampler = RandomSampler(train_subset, replacement=True)

batch_size = 384
train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    sampler=SequentialSampler(val_subset),
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
)

# %%
alpha = 0.25
gamma = 2.0


def validate(global_step, reward_weight):
    obs_loss_sum = 0.0
    reward_loss_sum = 0.0
    token_cnt = 0
    tp = pos = fp = 0

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
            B, T, C, Ht, Wt = frames.shape

            frames = frames.view(-1, 3, 84, 84).to(
                device, dtype=torch.float32, non_blocking=True
            )
            actions = actions.to(device, dtype=torch.long, non_blocking=True)
            rewards = rewards.to(
                device, dtype=torch.float32, non_blocking=True
            )
            game_ids = game_ids.to(device)

            with autocast(device_type="cuda", enabled=is_amp):
                # 移动到 autocast 内
                z_e = encoder(frames)
                _, indices, _ = quantizer(z_e)
                obs_tokens = indices.view(B, T, 16)

                pred_obs_logits, pred_rewards = world_model(
                    obs_tokens, actions, game_ids
                )

                obs_loss = F.cross_entropy(
                    pred_obs_logits.reshape(-1, 512),  # (B*T*16, 512)
                    obs_tokens.reshape(-1),  # (B*T*16,)
                    reduction="mean",
                )
                reward_target = (rewards.abs() > 1e-6).float()
                reward_loss = sigmoid_focal_loss(
                    pred_rewards,
                    reward_target,
                    reduction="mean",
                    alpha=alpha,
                    gamma=gamma,
                )

            batch_tokens = obs_tokens.numel()
            obs_loss_sum += obs_loss.item() * batch_tokens
            reward_loss_sum += reward_loss.item() * batch_tokens
            token_cnt += batch_tokens

            # 分组计算 losses 和 tp/fp
            unique_gids = game_ids.unique()
            for g in unique_gids:
                mask = game_ids == g
                if mask.sum() == 0:
                    continue

                # Slice
                obs_tokens_g = obs_tokens[mask]
                pred_obs_logits_g = pred_obs_logits[mask]
                rewards_g = rewards[mask]
                pred_rewards_g = pred_rewards[mask]

                # Group losses (用全局 gamma)
                obs_loss_g = F.cross_entropy(
                    pred_obs_logits_g.reshape(-1, 512),
                    obs_tokens_g.reshape(-1),
                    reduction="mean",
                )
                reward_target_g = (rewards_g.abs() > 1e-6).float()
                reward_loss_g = sigmoid_focal_loss(
                    pred_rewards_g,
                    reward_target_g,
                    reduction="mean",
                    alpha=alpha,
                    gamma=gamma,
                )

                tokens_g = obs_tokens_g.numel()
                gid = g.item()
                val_game_obs_loss_sum[gid] += obs_loss_g.item() * tokens_g
                val_game_reward_loss_sum[gid] += (
                    reward_loss_g.item() * tokens_g
                )
                val_game_token_cnt[gid] += tokens_g

                # 分组计算 tp/fp (替换原 per-sample)
                prob_g = torch.sigmoid(pred_rewards_g)
                pred_g = prob_g > 0.5
                target_g = reward_target_g.bool()  # 用 bool 以匹配
                val_game_tp[gid] += (pred_g & target_g).sum().item()
                val_game_pos[gid] += target_g.sum().item()
                val_game_fp[gid] += (pred_g & (~target_g)).sum().item()

            # 整体 tp/fp (保持)

            prob = torch.sigmoid(pred_rewards)
            pred = prob > 0.5
            tp += (pred & reward_target.bool()).sum().item()
            pos += reward_target.sum().item()
            fp += (pred & (~reward_target.bool())).sum().item()

        # 移动到循环外：计算 avg 和 log
        for gid in range(6):
            if val_game_token_cnt[gid] > 0:
                obs_loss_game = (
                    val_game_obs_loss_sum[gid] / val_game_token_cnt[gid]
                )
                reward_loss_game = (
                    val_game_reward_loss_sum[gid] / val_game_token_cnt[gid]
                )
                weighted_reward_loss_game = reward_loss_game * reward_weight
                total_loss_game = obs_loss_game + weighted_reward_loss_game
                recall_game = (
                    val_game_tp[gid] / val_game_pos[gid]
                    if val_game_pos[gid]
                    else float("nan")
                )
                precision_game = (
                    val_game_tp[gid] / (val_game_tp[gid] + val_game_fp[gid])
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
                        f"val/game_{games[gid]}/raw_reward_loss": reward_loss_game,  # 区分 raw
                        f"val/game_{games[gid]}/weighted_reward_loss": weighted_reward_loss_game,
                        f"val/game_{games[gid]}/total_loss": total_loss_game,
                        f"val/game_{games[gid]}/recall": recall_game,
                        f"val/game_{games[gid]}/precision": precision_game,
                        f"val/game_{games[gid]}/f1": f1_game,
                    },
                    step=global_step,
                )

        obs_loss_epoch = obs_loss_sum / token_cnt
        reward_loss_epoch = reward_loss_sum / token_cnt
        weighted_reward_loss_epoch = reward_loss_epoch * reward_weight
        total_loss_epoch = obs_loss_epoch + weighted_reward_loss_epoch

        recall = tp / pos if pos else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (
            (2 * precision * recall / (precision + recall))
            if precision and recall
            else float("nan")
        )

        wandb.log(
            {
                "all_val/obs_loss": obs_loss_epoch,
                "all_val/raw_reward_loss": reward_loss_epoch,  # 区分
                "all_val/weighted_reward_loss": weighted_reward_loss_epoch,
                "all_val/total_loss": total_loss_epoch,
                "all_val/recall": recall,
                "all_val/precision": precision,
                "all_val/f1": f1,
            },
            step=global_step,
        )

        # scheduler.step(reward_loss_epoch)

    return total_loss_epoch


# %%
best_val_loss = float("inf")
checkpoint_dir = "checkpoints_new"
os.makedirs(checkpoint_dir, exist_ok=True)

patience = 5  # 连续5次val不改善就stop
early_stop_counter = 0
min_delta = 0.0001  # 最小改善阈值

wandb.watch(world_model, log="gradients", log_freq=1000)
scaler = GradScaler(enabled=is_amp)
global_step = 0

initial_reward_weight = 1.0  # 初始值，根据日志调优
decay_rate = 1.0  # 每 epoch 衰减率
min_reward_weight = 1.0  # 最小值，防止过低

for epoch in range(1000):
    bar = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1:02d}")
    world_model.train()
    for frames, actions, rewards, game_ids in bar:
        global_step += 1

        B, T, C, Ht, Wt = frames.shape

        frames = frames.view(-1, 3, 84, 84).to(
            device, dtype=torch.float32, non_blocking=True
        )
        actions = actions.to(device, dtype=torch.long, non_blocking=True)
        rewards = rewards.to(device, dtype=torch.float32, non_blocking=True)
        game_ids = game_ids.to(device)

        with torch.no_grad():
            z_e = encoder(frames)
            _, indices, _ = quantizer(z_e)
            obs_tokens = indices.view(B, T, 16)

        with autocast(device_type="cuda", enabled=is_amp):
            pred_obs_logits, pred_rewards = world_model(
                obs_tokens, actions, game_ids
            )

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
                alpha=alpha,
                gamma=gamma,
            )

            reward_weight = max(
                initial_reward_weight * (decay_rate**epoch), min_reward_weight
            )
            weighted_reward_loss = reward_loss * reward_weight
            loss = obs_loss + weighted_reward_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            world_model.parameters(), max_norm=1.0, norm_type=2.0
        )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(
            {
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/obs_loss": obs_loss.item(),
                "train/reward_loss": weighted_reward_loss.item(),
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
            },
            step=global_step,
        )

        bar.set_postfix(
            obs_loss=obs_loss.item(),
            reward_loss=weighted_reward_loss.item(),
            loss=loss.item(),
        )

    total_loss_epoch = validate(global_step, reward_weight)

    if total_loss_epoch < best_val_loss - min_delta:
        best_val_loss = total_loss_epoch
        early_stop_counter = 0
        wm_path = os.path.join(
            checkpoint_dir, f"best_model_step_{global_step}.safetensors"
        )
        te_path = os.path.join(
            checkpoint_dir, f"task_embed_step_{global_step}.safetensors"
        )

        save_file(world_model.state_dict(), wm_path)
        save_file(task_embed.state_dict(), te_path)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at step {global_step}")
            break

run.finish()

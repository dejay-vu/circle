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
amp_dtype = torch.float16
horizon = 8

encoder_state = load_file("pretrained/encoder.safetensors")
vq_state = load_file("pretrained/vq.safetensors")
decoder_state = load_file("pretrained/decoder.safetensors")


# %%
class MultiTrajectoryDataset(Dataset):

    def __init__(self, games, horizon):
        self.root = zarr.open_group("dataset50k.zarr", mode="r")
        self.games = games
        self.horizon = horizon
        self.data = []
        self.reward_normalizers = []

        for gid, game in enumerate(self.games):
            frames = self.root[game]["frames"][:]
            actions = self.root[game]["actions"][:]
            rewards = self.root[game]["rewards"][:]
            dones = self.root[game]["dones"][:]

            # number of dones
            print(f"Game {game} has {dones.sum()} dones")

            # number of non-zero rewards
            print(
                f"Game {game} has {len(rewards[rewards != 0])} non-zero rewards"
            )

            max_reward = np.abs(rewards).max() if np.any(rewards) else 1.0
            self.reward_normalizers.append(max_reward)

            for idx in trange(len(frames) - horizon - 1):
                end = idx + horizon
                done_pos = np.where(dones[idx:end] == True)[0]
                if len(done_pos) > 0:
                    end = idx + done_pos[0] + 1  # include the done frame

                frame_seq = frames[idx : end + 1, -1]  # (H+1, 3, 84, 84)
                action_seq = actions[idx:end]  # (H,)
                reward_seq = rewards[idx:end]  # (H,)
                done_seq = dones[idx:end]  # (H,)

                seq_len = end - idx
                mask_seq = np.zeros(horizon, dtype=bool)
                mask_seq[:seq_len] = True  # (H,)

                if seq_len < horizon:
                    pad_len = horizon - seq_len
                    frame_seq = np.pad(
                        frame_seq,
                        ((0, pad_len), (0, 0), (0, 0), (0, 0)),
                        mode="constant",
                        constant_values=0,
                    )
                    action_seq = np.pad(
                        action_seq,
                        (0, pad_len),
                        mode="constant",
                        constant_values=18,
                    )
                    reward_seq = np.pad(
                        reward_seq,
                        (0, pad_len),
                        mode="constant",
                        constant_values=0.0,
                    )
                    done_seq = np.pad(
                        done_seq,
                        (0, pad_len),
                        mode="constant",
                        constant_values=False,
                    )

                self.data.append(
                    (
                        frame_seq,
                        action_seq,
                        reward_seq,
                        done_seq,
                        mask_seq,
                        gid,
                    )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_seq, action_seq, reward_seq, done_seq, mask_seq, gid = self.data[
            idx
        ]

        # Normalize rewards
        max_reward = self.reward_normalizers[gid]
        reward_seq = torch.from_numpy(reward_seq).float()
        reward_seq = torch.clamp(reward_seq / max_reward, -1.0, 1.0)

        return (
            torch.from_numpy(frame_seq).float().div_(255),  # (H, 3, 84, 84)
            torch.from_numpy(action_seq).long(),  # (H,)
            reward_seq,  # (H,)
            torch.from_numpy(done_seq).bool(),  # (H,)
            torch.from_numpy(mask_seq).bool(),  # (H,)
            torch.tensor(gid, dtype=torch.long),  # game id
        )


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Task embedding
task_embed = TaskEmbed(num_known_tasks=6).to(device)

# Initialize models
world_model = MiniWorldModel(task_embed=task_embed).to(device)

# Load pretrained weights
encoder = Encoder().to(device)
quantizer = vq_vae.to(device)
encoder.load_state_dict(encoder_state)
quantizer.load_state_dict(vq_state)

# world_model = torch.compile(world_model, mode="default")
world_model.train()
encoder.eval()
quantizer.eval()

base_lr = 1e-4
obs_head_lr = 3e-4
reward_head_lr = 1e-5
done_head_lr = 1e-5
weight_decay = 1e-2
optimizer = torch.optim.AdamW(
    [
        {
            "params": (
                list(world_model.transformer.parameters())
                + list(world_model.task_embed.parameters())
            ),
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": (list(world_model.obs_head.parameters())),
            "lr": obs_head_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": (list(world_model.reward_head.parameters())),
            "lr": reward_head_lr,
            "weight_decay": 0.0,
        },
        {
            "params": (list(world_model.done_head.parameters())),
            "lr": done_head_lr,
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
dataset = MultiTrajectoryDataset(games, horizon=horizon)

rng = np.random.default_rng(seed=42)
indices = np.arange(len(dataset))
perm = rng.permutation(indices)
split = int(len(perm) * 0.9)
train_indices = perm[:split]
val_indices = perm[split:]

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

train_sampler = RandomSampler(train_subset)

batch_size = 512
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
@torch.no_grad()
def preprocess_batch(
    frames,
    actions,
    rewards,
    dones,
    masks,
    game_ids,
    device,
):
    B, T = actions.shape

    frames = frames.reshape(-1, 3, 84, 84).to(device, non_blocking=True)
    actions = actions.to(device, non_blocking=True)
    rewards = rewards.to(device, non_blocking=True)
    dones = dones.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    game_ids = game_ids.to(device, non_blocking=True)

    with autocast(device_type="cuda", dtype=amp_dtype, enabled=is_amp):
        z_e = encoder(frames)
        _, indices, _ = quantizer(z_e)
        indices = indices.reshape(B, T + 1, 16)  # (B, T+1, 16)
        curr_obs_tokens = indices[:, :-1]  # (B, T, 16)
        next_obs_tokens = indices[:, 1:]  # (B, T, 16)

    return (
        curr_obs_tokens,
        next_obs_tokens,
        actions,
        rewards,
        dones,
        masks,
        game_ids,
    )


reward_weight = 20.0


def compute_loss(
    world_model,
    curr_obs_tokens,
    next_obs_tokens,
    actions,
    rewards,
    dones,
    obs_masks,
    seq_masks,
    game_ids,
):
    pred_next_obs_tokens_logits, pred_reward, pred_done = world_model(
        curr_obs_tokens, actions, game_ids
    )  # (B, T, 16, 512), (B, T), (B, T)

    obs_loss = F.cross_entropy(
        pred_next_obs_tokens_logits.reshape(-1, 512),  # (B*T*16, 512)
        next_obs_tokens.reshape(-1),  # (B*T*16,)
        reduction="none",
    )  # (B*T*16,)
    obs_loss = obs_loss[obs_masks].mean()

    weights = torch.ones_like(rewards)
    weights[rewards.abs() > 1e-6] = reward_weight
    reward_loss = F.huber_loss(
        pred_reward.reshape(-1),  # (B*T,)
        rewards.reshape(-1),  # (B*T,)
        reduction="none",
        weight=weights.reshape(-1),
    )
    reward_loss = reward_loss[seq_masks].mean()

    done_target = dones.float()
    done_loss = sigmoid_focal_loss(
        pred_done.reshape(-1),  # (B*T,)
        done_target.reshape(-1),  # (B*T,)
        reduction="none",
        alpha=alpha,
        gamma=gamma,
    )
    done_loss = done_loss[seq_masks].mean()

    return (
        obs_loss,
        reward_loss,
        done_loss,
        done_target,
        pred_done,
    )


def compute_prf(
    tp: int,
    fp: int,
    pos: int,
    *,
    nan_when_no_pos=True,
    nan_when_no_pred_pos=False,
    eps=1e-12,
):
    pred_pos = tp + fp

    if pos == 0:
        if nan_when_no_pos:
            return dict(
                precision=float("nan"), recall=float("nan"), f1=float("nan")
            )
        else:
            # 定义为 0
            return dict(precision=0.0, recall=float("nan"), f1=float("nan"))

    if pred_pos == 0:
        # 没预测正
        if nan_when_no_pred_pos:
            return dict(precision=float("nan"), recall=0.0, f1=float("nan"))
        else:
            return dict(precision=0.0, recall=0.0, f1=0.0)

    precision = tp / (pred_pos + eps)
    recall = tp / (pos + eps)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


# %%
alpha = 0.75
gamma = 2.0


@torch.inference_mode()
def validate(global_step):
    world_model.eval()

    obs_loss_sum = 0.0
    reward_loss_sum = 0.0
    done_loss_sum = 0.0

    valid_obs_cnt = 0
    valid_steps_cnt = 0

    reward_tp = reward_fp = reward_pos = 0
    done_tp = done_fp = done_pos = 0

    val_bar = tqdm(
        val_loader,
        leave=False,
        desc=f"Validating at step {global_step}",
    )

    for frames, actions, rewards, dones, masks, game_ids in val_bar:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=is_amp):
            (
                curr_obs_tokens,
                next_obs_tokens,
                actions,
                rewards,
                dones,
                masks,
                game_ids,
            ) = preprocess_batch(
                frames, actions, rewards, dones, masks, game_ids, device
            )

            flat_masks = masks.reshape(-1)  # (B*T,)
            repeated_masks = torch.repeat_interleave(
                flat_masks, 16
            )  # (B*T*16,)

            (
                obs_loss,
                reward_loss,
                done_loss,
                done_target,
                pred_done,
            ) = compute_loss(
                world_model,
                curr_obs_tokens,
                next_obs_tokens,
                actions,
                rewards,
                dones,
                obs_masks=repeated_masks,
                seq_masks=flat_masks,
                game_ids=game_ids,
            )

        num_valid_obs_tokens = repeated_masks.sum().item()
        num_valid_steps = flat_masks.sum().item()
        valid_obs_cnt += num_valid_obs_tokens
        valid_steps_cnt += num_valid_steps
        obs_loss_sum += obs_loss.item() * num_valid_obs_tokens
        reward_loss_sum += reward_loss.item() * num_valid_steps
        done_loss_sum += done_loss.item() * num_valid_steps

        done_prob = torch.sigmoid(pred_done)
        done_pred = done_prob > 0.5
        done_tp += (done_pred & done_target.bool()).sum().item()
        done_pos += done_target.sum().item()
        done_fp += (done_pred & (~done_target.bool())).sum().item()

    obs_loss_epoch = obs_loss_sum / valid_obs_cnt if valid_obs_cnt > 0 else 0
    reward_loss_epoch = (
        reward_loss_sum / valid_steps_cnt if valid_steps_cnt > 0 else 0
    )
    done_loss_epoch = (
        done_loss_sum / valid_steps_cnt if valid_steps_cnt > 0 else 0
    )

    total_loss_epoch = obs_loss_epoch + reward_loss_epoch + done_loss_epoch

    done_precision, done_recall, done_f1 = compute_prf(
        done_tp,
        done_fp,
        done_pos,
        nan_when_no_pos=False,
        nan_when_no_pred_pos=True,
    )

    wandb.log(
        {
            "val/obs_loss": obs_loss_epoch,
            "val/reward_loss": reward_loss_epoch,
            "val/done_loss": done_loss_epoch,
            "val/total_loss": total_loss_epoch,
            "val/done-recall": done_recall,
            "val/done-precision": done_precision,
            "val/done-f1": done_f1,
        },
        step=global_step,
    )

    # scheduler.step(reward_loss_epoch)

    return total_loss_epoch


# %%
if __name__ == "__main__":
    best_val_loss = float("inf")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    scaler = GradScaler(enabled=is_amp)

    patience = 5
    early_stop_counter = 0
    min_delta = 0.0001

    run = wandb.init(
        project="pretrain-world-model",
        name="Multigames-50k",
        config={
            "batch_size": batch_size,
            "base_lr": base_lr,
            "obs_head_lr": obs_head_lr,
            "reward_head_lr": reward_head_lr,
            "done_head_lr": done_head_lr,
            "weight_decay": weight_decay,
            "horizon": horizon,
            "reward_weight": reward_weight,
            "alpha": alpha,
            "gamma": gamma,
        },
    )
    wandb.watch(world_model, log="all", log_freq=1000)

    global_step = 0
    for epoch in range(1000):
        bar = tqdm(train_loader, leave=True, desc=f"Epoch {epoch + 1:02d}")
        world_model.train()
        for frames, actions, rewards, dones, masks, game_ids in bar:
            global_step += 1

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=is_amp):
                (
                    curr_obs_tokens,
                    next_obs_tokens,
                    actions,
                    rewards,
                    dones,
                    masks,
                    game_ids,
                ) = preprocess_batch(
                    frames, actions, rewards, dones, masks, game_ids, device
                )

                flat_masks = masks.reshape(-1)  # (B*T,)
                repeated_masks = torch.repeat_interleave(
                    flat_masks, 16
                )  # (B*T*16,)

                (
                    obs_loss,
                    reward_loss,
                    done_loss,
                    done_target,
                    pred_done,
                ) = compute_loss(
                    world_model,
                    curr_obs_tokens,
                    next_obs_tokens,
                    actions,
                    rewards,
                    dones,
                    obs_masks=repeated_masks,
                    seq_masks=flat_masks,
                    game_ids=game_ids,
                )

            loss = obs_loss + reward_loss + done_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            done_prob = torch.sigmoid(pred_done)
            done_pred = done_prob > 0.5
            done_tp = (done_pred & done_target.bool()).sum().item()
            done_pos = done_target.sum().item()
            done_fp = (done_pred & (~done_target.bool())).sum().item()

            done_precision, done_recall, done_f1 = compute_prf(
                done_tp,
                done_fp,
                done_pos,
                nan_when_no_pos=False,
                nan_when_no_pred_pos=True,
            )

            wandb.log(
                {
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/obs_loss": obs_loss.item(),
                    "train/reward_loss": reward_loss.item(),
                    "train/done_loss": done_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/done-recall": done_recall,
                    "train/done-precision": done_precision,
                    "train/done-f1": done_f1,
                },
                step=global_step,
            )

            bar.set_postfix(
                obs_loss=obs_loss.item(),
                reward_loss=reward_loss.item(),
                done_loss=done_loss.item(),
                loss=loss.item(),
            )

        total_loss_epoch = validate(global_step)

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

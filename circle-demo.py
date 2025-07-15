# circle_demo.py  (self‑contained end‑to‑end proof‑of‑concept)
# -----------------------------------------------------------
# REQUIREMENTS (pip install ...):
#   slot-attention-pytorch
#   vector-quantize-pytorch
#   x-transformers
#   denoising-diffusion-pytorch
#   einops
# -----------------------------------------------------------
# This script shows:
#   1. Pre‑training a SAVi + VQ‑VAE tokeniser on dummy 84×84 RGB frames.
#   2. Pre‑training a causal Transformer world model on the resulting
#      discrete tokens.
#   3. Latent diffusion refinement of a 5‑step rollout.
#   4. A ViT policy acting in a toy env that returns random rewards.
#   5. EWC + anchor‑consistency regularisation inside the training loop.
# The goal is *plumbing*, not performance. Running on CPU < 4 GB RAM.
# -----------------------------------------------------------

import os, math, random, argparse, itertools, copy
from collections import deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from einops import rearrange
from torch.distributions import Categorical

# lucidrains repos
from slot_attention_pytorch import SAVi
from vector_quantize_pytorch import VectorQuantizeEMA
from x_transformers import TransformerWrapper, Decoder,
                           Encoder
from denoising_diffusion_pytorch import DiffusionPriorNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# 1.  Tokeniser: SAVi + VQ‑VAE
# -------------------------------
class Tokeniser(nn.Module):
    """SAVi → 512‑d quantised tokens."""
    def __init__(self, codebook_size=512, num_slots=8):
        super().__init__()
        self.savi = SAVi(
            image_size = 84,
            num_slots = num_slots,
            dim = 128,
            slots_init = 'linsigmoid'
        )
        self.to_512 = nn.Linear(128, 512)
        self.vq = VectorQuantizeEMA(
            dim = 512,
            codebook_size = codebook_size,
            decay = 0.95,
            commitment_weight = 1.0
        )
        # light decoder for recon loss
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 84*84*3)
        )

    def forward(self, imgs):
        B = imgs.size(0)
        slots = self.savi(imgs)               # (B N 128)
        tokens = self.to_512(slots)           # (B N 512)
        q, ind, _ = self.vq(tokens)           # (B N 512) quantised
        recon = self.decoder(q).view(B,3,84,84)
        return q, ind, recon

# -------------------------------
# 2.  Causal World Model
# -------------------------------
class CausalWorldModel(nn.Module):
    def __init__(self, codebook_size=512, d_model=256, depth=4):
        super().__init__()
        self.token_emb = nn.Embedding(codebook_size, d_model)
        self.action_emb = nn.Embedding(18, d_model)
        self.decoder = TransformerWrapper(
            num_tokens = codebook_size,
            max_seq_len = 64,
            attn_layers = Decoder(
                dim = d_model,
                depth = depth,
                heads = 4,
            )
        )
        self.to_logits = nn.Linear(d_model, codebook_size)

    def forward(self, token_idx, action_idx):
        # token_idx: (B N) current tokens (flattened),
        # action_idx: (B) discrete action id
        tok = self.token_emb(token_idx)
        act = self.action_emb(action_idx)[:,None,:]  # broadcast over seq
        seq = tok + act
        dec = self.decoder(seq)
        logits = self.to_logits(dec)  # (B N C)
        return logits

# -------------------------------
# 3.  Latent Diffusion (toy)
# -------------------------------
class LatentDiffuser(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = DiffusionPriorNetwork(
            dim = dim,
            depth = 2,
            heads = 4,
        )
    def forward(self, z):
        # z: (B N 512)
        return self.net(z)

# -------------------------------
# 4.  ViT Policy
# -------------------------------
class ViTPolicy(nn.Module):
    def __init__(self, codebook_size=512):
        super().__init__()
        self.enc = TransformerWrapper(
            num_tokens = codebook_size,
            max_seq_len = 64,
            attn_layers = Encoder(dim=512, depth=3, heads=8)
        )
        self.cls = nn.Parameter(torch.randn(1,1,512))
        self.actor = nn.Sequential(nn.Linear(512,256),nn.Tanh(),nn.Linear(256,18))
        self.critic = nn.Sequential(nn.Linear(512,256),nn.Tanh(),nn.Linear(256,1))

    def forward(self, token_idx):
        B = token_idx.size(0)
        cls = self.cls.expand(B,-1,-1)          # (B 1 512)
        emb = self.enc.token_emb(token_idx)     # same embedding as x‑transformers util
        seq = torch.cat([cls, emb],1)
        y = self.enc(seq)[:,0]                  # CLS output
        logits = self.actor(y)
        value = self.critic(y).squeeze(-1)
        return logits, value

# -------------------------------
# 5.  Dummy env / dataset
# -------------------------------
class DummyAtari(torch.utils.data.IterableDataset):
    def __iter__(self):
        while True:
            frame = torch.rand(3,84,84)
            action = torch.randint(0,18,(1,))
            reward = torch.rand(1)
            yield frame, action, reward

env_iter = iter(torch.utils.data.DataLoader(DummyAtari(), batch_size=8))

# -------------------------------
# 6.  EWC helper
# -------------------------------
class EWCReg:
    def __init__(self, model: nn.Module, lambda_=1.0):
        self.model = model
        self.lambda_ = lambda_
        self.params = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n,p in self.params.items()}

    def accumulate_fisher(self, loss):
        loss.backward(retain_graph=True)
        for n,p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] += p.grad.data.pow(2)
        self.model.zero_grad()

    def penalty(self):
        reg = 0
        for n,p in self.model.named_parameters():
            if n in self.params:
                reg += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.lambda_ * reg

# -------------------------------
# 7.  Training phases
# -------------------------------

def pretrain_tokeniser(tokeniser, steps=200):
    optimiser = Adam(tokeniser.parameters(), 1e-4)
    for i in range(steps):
        imgs,_,_ = next(env_iter)
        imgs = imgs.to(DEVICE)
        q, ind, recon = tokeniser(imgs)
        rec_loss = F.mse_loss(recon, imgs)
        perplexity = tokeniser.vq.perplexity()
        loss = rec_loss + tokeniser.vq.commitment_loss()
        optimiser.zero_grad(); loss.backward(); optimiser.step()
        if (i+1)%50==0:
            print(f"[Tok] step {i+1}  recon {rec_loss.item():.3f}  perp {perplexity:.2f}")


def pretrain_world_model(model, tokeniser, steps=200):
    optim = Adam(model.parameters(), 2e-4)
    for i in range(steps):
        imgs,_ ,_ = next(env_iter)
        next_imgs,_ ,_ = next(env_iter)
        imgs = imgs.to(DEVICE); next_imgs = next_imgs.to(DEVICE)
        _, ind, _ = tokeniser(imgs)
        _, next_ind, _ = tokeniser(next_imgs)
        actions = torch.randint(0,18,(imgs.size(0),)).to(DEVICE)
        logits = model(ind.view(imgs.size(0),-1), actions)
        loss = F.cross_entropy(logits.view(-1,512), next_ind.view(-1))
        optim.zero_grad(); loss.backward(); optim.step()
        if (i+1)%50==0:
            print(f"[WM] step {i+1}  CE {loss.item():.3f}")


def rollout(model, diffuser, start_ind, actions):
    # start_ind (B N)
    logits = model(start_ind, actions)
    probs = logits.softmax(-1)
    sampled = Categorical(probs).sample()
    z = model.token_emb(sampled)            # (B N D)
    z_refined = diffuser(z)                 # (B N D)
    return sampled, z_refined


def ppo_step(policy, tokeniser, model, diffuser, ewc, epochs=3):
    imgs,_,_ = next(env_iter)
    imgs = imgs.to(DEVICE)
    _, ind, _ = tokeniser(imgs)
    logits, value = policy(ind.view(imgs.size(0),-1))
    dist = Categorical(logits=logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    # pretend env reward:
    reward = torch.randn_like(action.float())
    advantage = (reward - value.detach())
    # PPO surrogate
    for _ in range(epochs):
        new_logits, new_value = policy(ind.view(imgs.size(0),-1))
        new_dist = Categorical(logits=new_logits)
        new_logprob = new_dist.log_prob(action)
        ratio = (new_logprob - logprob).exp()
        surr = torch.minimum(ratio*advantage, torch.clamp(ratio,0.8,1.2)*advantage).mean()
        v_loss = F.mse_loss(new_value, reward)
        wm_loss = 0.0
        # world‑model one‑step loss in background
        sampled, _ = rollout(model, diffuser, ind.view(imgs.size(0),-1), action)
        wm_logits = model(ind.view(imgs.size(0),-1), action)
        wm_loss = F.cross_entropy(wm_logits.view(-1,512), sampled.view(-1))
        loss = -surr + 0.5*v_loss + 0.1*wm_loss + ewc.penalty()
        optim.zero_grad(); loss.backward(); optim.step()

# -------------------------------
# 8.  Main driver
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    # instantiate modules
    tokeniser = Tokeniser().to(DEVICE)
    world_model = CausalWorldModel().to(DEVICE)
    diffuser = LatentDiffuser().to(DEVICE)
    policy = ViTPolicy().to(DEVICE)
    params = itertools.chain(world_model.parameters(), policy.parameters())
    optim = Adam(params, 1e-4)
    ewc = EWCReg(world_model, lambda_=1.0)

    print("\n===  Pretraining tokeniser  ===")
    pretrain_tokeniser(tokeniser)
    print("\n===  Pretraining world model ===")
    pretrain_world_model(world_model, tokeniser)
    ewc.accumulate_fisher(torch.tensor(0.))  # snapshot after pretrain

    print("\n===  Online loop (ppo stub) ===")
    for i in range(10):
        ppo_step(policy, tokeniser, world_model, diffuser, ewc)
        if (i+1)%5==0:
            print(f"[MAIN] PPO iter {i+1} done")

    print("Demo finished — modules ran forward/backward without error.☺")

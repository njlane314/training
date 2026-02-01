import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import MinkowskiEngine as ME

from . import config as cfg
from .data import BalancedBatchSampler, ShardDataset, collate
from .model import MinkUNetClassifier


def stratified_split(labels, frac, seed):
    """
    @brief Split indices into stratified train/validation sets.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(labels.shape[0], dtype=np.int64)
    sig = idx[labels == 1]
    bkg = idx[labels == 0]
    rng.shuffle(sig)
    rng.shuffle(bkg)
    ns = int(round(sig.size * frac))
    nb = int(round(bkg.size * frac))
    val = np.concatenate([sig[:ns], bkg[:nb]])
    trn = np.concatenate([sig[ns:], bkg[nb:]])
    rng.shuffle(val)
    rng.shuffle(trn)
    return trn, val


def sample_probe_indices(labels, batch, rng):
    """
    @brief Select a random probe batch for validation monitoring.
    """
    h = batch // 2
    sig = np.where(labels == 1)[0]
    bkg = np.where(labels == 0)[0]
    if sig.size < h or bkg.size < h:
        raise ValueError("not enough events for probe batch")
    s = rng.choice(sig, size=h, replace=False)
    b = rng.choice(bkg, size=h, replace=False)
    p = np.concatenate([s, b]).astype(np.int64, copy=False)
    rng.shuffle(p)
    return p


def cosine_warmup(optimizer, total_steps: int, warmup_ratio: float = 0.02, min_lr_ratio: float = 0.05):
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(1, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * t))

    return LambdaLR(optimizer, lr_lambda)


def attach_grad_hooks(model: nn.Module):
    handles = []

    def bwd_hook(mod, grad_input, grad_output):
        # grad_output is a tuple; take first tensor-like entry if present
        go = None
        if isinstance(grad_output, (tuple, list)) and len(grad_output) > 0:
            go = grad_output[0]
        if go is None or not torch.is_tensor(go):
            return
        g = go.detach()
        gnorm = float(g.float().norm().item())
        gmax = float(g.float().abs().max().item())
        print(f"    [bwd] {mod.__class__.__name__:<24} ‖g‖={gnorm:.3e} max|g|={gmax:.3e}", flush=True)

    for m in model.modules():
        # avoid very noisy hooks unless you want everything
        if isinstance(m, (ME.MinkowskiConvolution, ME.MinkowskiBatchNorm, ME.MinkowskiLinear)):
            handles.append(m.register_full_backward_hook(bwd_hook))

    return handles


def main():
    """
    @brief Train a MinkowskiEngine UNet classifier end-to-end.
    """
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.AMP and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    probe_every = max(1, int(getattr(cfg, "VAL_PROBE_EVERY", 1)))

    env_lr = os.environ.get("LR")
    print(f"Using LR={cfg.LR} (env LR={env_lr})", flush=True)

    meta = torch.load(os.path.join(cfg.SHARDS_DIR, "index.pt"), map_location="cpu")
    labels_all = np.asarray(meta["labels"], dtype=np.uint8)

    trn_idx, val_idx = stratified_split(labels_all, cfg.VAL_FRAC, cfg.SEED)
    trn_idx = np.sort(trn_idx)
    val_idx = np.sort(val_idx)

    trn_ds = ShardDataset(cfg.SHARDS_DIR, trn_idx)
    val_ds = ShardDataset(cfg.SHARDS_DIR, val_idx)

    trn_bs = BalancedBatchSampler(
        trn_ds.labels,
        trn_ds.shard_ids,
        trn_ds.local_ids,
        cfg.BATCH,
        cfg.SEED,
        steps=cfg.STEPS_PER_EPOCH,
    )
    trn_loader = torch.utils.data.DataLoader(
        trn_ds,
        batch_sampler=trn_bs,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.NUM_WORKERS > 0),
        multiprocessing_context="forkserver" if cfg.NUM_WORKERS > 0 else None,
        prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
        timeout=120 if cfg.NUM_WORKERS > 0 else 0,
        collate_fn=collate,
    )

    probe_rng = np.random.default_rng(cfg.SEED + 999)

    model = MinkUNetClassifier(in_channels=4, base=cfg.BASE_FILTERS, strides=cfg.NUM_STRIDES, dropout=cfg.DROPOUT).to(
        device
    )
    grad_handles = []
    if os.environ.get("DEBUG_GRADS", "0") != "0":
        grad_handles = attach_grad_hooks(model)
    # Weight-movement probes (simple, robust: linear head weights are always torch.Parameters)
    w_head0_0 = model.head[0].weight.detach().clone()
    w_head1_0 = model.head[-1].weight.detach().clone()
    print("\n--- Model Architecture (MinkUNet Classifier) ---")
    print(model)
    print("--------------------------------------------------")
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    accum = max(1, int(cfg.GRAD_ACCUM_STEPS))
    sched = None
    if getattr(cfg, "SCHED", False):
        total_steps = int(cfg.EPOCHS) * int(math.ceil(len(trn_loader) / accum))
        sched = cosine_warmup(
            opt,
            total_steps=total_steps,
            warmup_ratio=cfg.WARMUP_RATIO,
            min_lr_ratio=cfg.MIN_LR_RATIO,
        )
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    c, f, y = next(iter(trn_loader))
    t1 = time.time()
    print(f"warmup {t1-t0:.2f}s nnz={int(f.shape[0])} y_mean={float(y.mean()):.3f}", flush=True)
    print("\n--- Model Summary (torchinfo) ---", flush=True)
    try:
        import torchinfo
    except ModuleNotFoundError:
        print("torchinfo not installed; skipping model summary.", flush=True)
    else:
        try:
            summary = torchinfo.summary(
                model,
                input_data=(ME.SparseTensor(f, c, device=device),),
                verbose=0,
            )
            print(summary, flush=True)
        except Exception as exc:
            print(f"torchinfo summary failed: {exc}", flush=True)
    print("--------------------------------------------------", flush=True)

    best = float("inf")
    ema_t = None
    ema_v = None
    last_vl = None
    last_vacc = None
    step0 = 0
    micro_step = 0

    log_handle = None
    log_writer = None
    if cfg.LOG_OUT:
        needs_header = True
        if os.path.exists(cfg.LOG_OUT) and os.path.getsize(cfg.LOG_OUT) > 0:
            needs_header = False
        log_handle = open(cfg.LOG_OUT, "a", buffering=1, encoding="utf-8", newline="")
        log_writer = csv.writer(log_handle)
        if needs_header:
            log_writer.writerow(
                ["epoch", "step", "lr", "train", "val", "ema_train", "ema_val", "acc", "vacc"]
            )

    try:
        for epoch in range(cfg.EPOCHS):
            trn_bs.set_epoch(epoch)
            model.train()
            for i, (coords, feats, y) in enumerate(trn_loader):
                x = ME.SparseTensor(feats, coords, device=device)
                y = y.to(device, non_blocking=True).float().view(-1)

                if micro_step % accum == 0:
                    opt.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits = model(x).view(-1)
                    tloss = loss_fn(logits, y)
                    # If accumulating, backprop the mean gradient (keeps effective LR stable).
                    loss_to_bp = tloss / float(accum)
                if use_amp:
                    scaler.scale(loss_to_bp).backward()
                else:
                    loss_to_bp.backward()
                micro_step += 1
                stepped = False
                if micro_step % accum == 0:
                    if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
                        if use_amp:
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    if sched is not None:
                        sched.step()
                    stepped = True

                # Probe val only on optimizer steps (or first time), to reduce noise/overhead.
                do_probe = (last_vl is None) or (stepped and (step0 % probe_every == 0))
                if do_probe:
                    model.eval()
                    with torch.no_grad(), autocast(enabled=use_amp):
                        probe_local = sample_probe_indices(val_ds.labels, cfg.BATCH, probe_rng)
                        vb = val_ds.__getitems__(probe_local)
                        vcoords, vfeats, vy = collate(vb)
                        vprobe_x = ME.SparseTensor(vfeats, vcoords, device=device)
                        vprobe_y = vy.to(device, non_blocking=True).float().view(-1)
                        vlogits = model(vprobe_x).view(-1)
                        vloss = loss_fn(vlogits, vprobe_y)
                        vacc = (vlogits > 0).eq(vprobe_y > 0.5).float().mean().item()
                    model.train()
                    last_vl = float(vloss.item())
                    last_vacc = float(vacc)

                tl = float(tloss.item())
                vl = float(last_vl) if last_vl is not None else float("nan")
                ema_t = tl if ema_t is None else (cfg.EMA * ema_t + (1.0 - cfg.EMA) * tl)
                if do_probe:
                    ema_v = vl if ema_v is None else (cfg.EMA * ema_v + (1.0 - cfg.EMA) * vl)

                tacc = (logits > 0).eq(y > 0.5).float().mean().item()
                vacc = float(last_vacc) if last_vacc is not None else float("nan")
                lr = opt.param_groups[0]["lr"]
                if stepped:
                    step0 += 1
                step = step0
                line = (
                    f"{epoch+1:03d} {step:07d} lr={lr:.2e} train={tl:.6f} val={vl:.6f} "
                    f"ema={ema_t:.6f}/{ema_v:.6f} acc={tacc:.3f} vacc={vacc:.3f}"
                )
                print(line, flush=True)
                # Debug prints: logits/statistics + weight movement every 50 optimizer steps
                if stepped and (step % 50 == 0):
                    with torch.no_grad():
                        lf = logits.float()
                        l_mu = float(lf.mean().item())
                        l_sd = float(lf.std(unbiased=False).item())
                        p = torch.sigmoid(lf)
                        p_mu = float(p.mean().item())
                        p_sd = float(p.std(unbiased=False).item())
                        dw0 = float((model.head[0].weight.detach() - w_head0_0).abs().mean().item())
                        dw1 = float((model.head[-1].weight.detach() - w_head1_0).abs().mean().item())
                    print(
                        f"    logits μ/σ={l_mu:+.3e}/{l_sd:.3e}  p μ/σ={p_mu:.3f}/{p_sd:.3f}  "
                        f"ΔW_head0={dw0:.3e}  ΔW_head1={dw1:.3e}",
                        flush=True,
                    )
                if log_writer is not None:
                    log_writer.writerow(
                        [epoch + 1, step, lr, tl, vl, ema_t, ema_v, tacc, vacc]
                    )

            if micro_step % accum != 0:
                if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
                    if use_amp:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                if sched is not None:
                    sched.step()
                step0 += 1
                micro_step = 0

            if ema_v is not None and ema_v < best:
                best = float(ema_v)
                torch.save(
                    {
                        "epoch": int(epoch + 1),
                        "step": int(step0),
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict() if sched is not None else None,
                        "scaler": scaler.state_dict() if use_amp else None,
                        "best_ema_val": best,
                        "cfg": {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
                    },
                    cfg.OUT,
                )

        torch.save({"epoch": int(cfg.EPOCHS), "step": int(step0), "model": model.state_dict()}, cfg.OUT)
    finally:
        for handle in grad_handles:
            handle.remove()
        if log_handle is not None:
            log_handle.close()

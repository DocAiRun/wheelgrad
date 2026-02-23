"""
experiments/compare.py
──────────────────────
Side-by-side training comparison: Standard IEEE vs WheelGrad.

Intentionally uses a high learning rate to provoke instability —
the exact conditions where NaN crashes happen in practice.

Run:
    cd wheelgrad/
    python experiments/compare.py
    python experiments/compare.py --lr 0.5 --epochs 30 --seed 123
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.model import MiniTransformer, make_causal_mask, make_dataset

# ── ANSI ──────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    mask: torch.Tensor,
    batch_size: int = 32,
    inject_instability: bool = False,
    instability_step: int = 5,
) -> dict:
    """
    Train for one epoch. Returns metrics dict.
    
    inject_instability: at step `instability_step`, artificially inject
    a bad batch (all-same tokens → all-identical features → std=0 in LayerNorm)
    to trigger the exact 0/0 scenario in LayerNorm + uniform softmax.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    total_loss = 0.0
    nan_count  = 0
    grad_norms = []
    step_losses = []
    n_steps = 0

    idx = torch.randperm(x.size(0))
    x, y = x[idx], y[idx]

    for step, start in enumerate(range(0, x.size(0), batch_size)):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]

        # Inject instability: all-same tokens → LayerNorm std=0
        if inject_instability and step == instability_step:
            xb = torch.zeros_like(xb)  # all token 0 → identical embeddings
            yb = torch.zeros_like(yb)

        optimizer.zero_grad()

        logits = model(xb, mask)          # (B, T, vocab)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), yb.view(B * T))

        # Detect NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            step_losses.append(float('nan'))
            continue

        loss.backward()

        # Gradient norm (detect explosion)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        # Check for NaN gradients
        has_nan_grad = any(
            torch.isnan(p.grad).any().item()
            for p in model.parameters()
            if p.grad is not None
        )
        if has_nan_grad:
            nan_count += 1
            step_losses.append(float('nan'))
            optimizer.zero_grad()
            continue

        optimizer.step()
        total_loss  += loss.item()
        n_steps     += 1
        step_losses.append(loss.item())

    avg_loss  = total_loss / max(n_steps, 1)
    avg_gnorm = float(np.mean(grad_norms)) if grad_norms else 0.0

    return {
        'loss':       avg_loss,
        'nan_steps':  nan_count,
        'grad_norm':  avg_gnorm,
        'step_losses': step_losses,
    }


def run_experiment(
    mode: str,
    lr: float,
    epochs: int,
    seed: int,
    inject: bool,
    batch_size: int,
    quiet: bool = False,
) -> dict:
    """Run one full training experiment."""
    torch.manual_seed(seed)

    model = MiniTransformer(
        vocab_size=128, d_model=64, n_heads=4,
        n_layers=2, d_ff=128, max_seq_len=32,
        dropout=0.0, mode=mode
    )

    x, y = make_dataset(n_samples=512, seq_len=32, vocab_size=128, seed=seed)
    mask  = make_causal_mask(32)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'losses': [], 'nan_steps': [], 'grad_norms': [],
        'crashed_at': None, 'mode': mode
    }

    label  = f"{CYAN}WheelGrad{RESET}" if mode == 'wheel' else f"{YELLOW}Standard {RESET}"
    prefix = f"  [{label}]"

    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model, x, y, optimizer, mask,
            batch_size=batch_size,
            inject_instability=inject,
            instability_step=5,
        )

        history['losses'].append(metrics['loss'])
        history['nan_steps'].append(metrics['nan_steps'])
        history['grad_norms'].append(metrics['grad_norm'])

        loss_str = f"{metrics['loss']:.4f}" if not np.isnan(metrics['loss']) else f"{RED}NaN{RESET}"
        nan_str  = f"{RED}NaN steps: {metrics['nan_steps']}{RESET}" if metrics['nan_steps'] > 0 else f"{GREEN}clean{RESET}"
        gnorm    = f"{metrics['grad_norm']:.2f}"

        if not quiet:
            print(f"{prefix} Epoch {epoch:3d}/{epochs}  "
                  f"loss={loss_str}  grad_norm={gnorm}  {nan_str}")

        # Detect full crash (3+ consecutive NaN epochs)
        if len(history['losses']) >= 3 and all(
            np.isnan(l) for l in history['losses'][-3:]
        ):
            if history['crashed_at'] is None:
                history['crashed_at'] = epoch
            if not quiet:
                print(f"{prefix} {RED}⚠ Training crashed — NaN for 3 consecutive epochs{RESET}")
            # Fill remaining epochs as NaN
            for _ in range(epoch + 1, epochs + 1):
                history['losses'].append(float('nan'))
                history['nan_steps'].append(0)
                history['grad_norms'].append(0.0)
            break

    return history


# ── Report ────────────────────────────────────────────────────────────────────

def print_summary(std_hist: dict, wheel_hist: dict, lr: float, inject: bool):
    print(f"\n{BOLD}{'═' * 62}{RESET}")
    print(f"{BOLD}  WHEELGRAD vs STANDARD — TRAINING STABILITY REPORT{RESET}")
    print(f"{'─' * 62}")
    print(f"  Learning rate : {lr}")
    print(f"  Instability   : {'ENABLED (all-zero batch injection)' if inject else 'disabled'}")
    print(f"{'─' * 62}")

    def fmt_loss(h):
        valid = [l for l in h['losses'] if not np.isnan(l)]
        if not valid:
            return f"{RED}ALL NaN{RESET}"
        return f"{valid[-1]:.4f}"

    def fmt_crash(h):
        if h['crashed_at']:
            return f"{RED}epoch {h['crashed_at']}{RESET}"
        return f"{GREEN}never crashed{RESET}"

    std_nan_total   = sum(std_hist['nan_steps'])
    wheel_nan_total = sum(wheel_hist['nan_steps'])
    std_valid_epochs   = sum(1 for l in std_hist['losses'] if not np.isnan(l))
    wheel_valid_epochs = sum(1 for l in wheel_hist['losses'] if not np.isnan(l))

    print(f"  {'Metric':<30} {'Standard':<18} {'WheelGrad'}")
    print(f"  {'─' * 58}")
    print(f"  {'Final loss':<30} {fmt_loss(std_hist):<28} {fmt_loss(wheel_hist)}")
    print(f"  {'Crashed at':<30} {fmt_crash(std_hist):<28} {fmt_crash(wheel_hist)}")
    print(f"  {'NaN steps total':<30} {RED}{std_nan_total}{RESET:<27} {GREEN}{wheel_nan_total}{RESET}")
    print(f"  {'Valid epochs':<30} {std_valid_epochs:<18} {wheel_valid_epochs}")

    # Verdict
    print(f"\n{'─' * 62}")
    if wheel_nan_total < std_nan_total or (
        wheel_hist['crashed_at'] is None and std_hist['crashed_at'] is not None
    ):
        print(f"  {GREEN}{BOLD}✓ WheelGrad is more stable than Standard IEEE 754{RESET}")
        if std_hist['crashed_at']:
            print(f"  {GREEN}  Standard crashed at epoch {std_hist['crashed_at']} — WheelGrad continued{RESET}")
    elif wheel_nan_total == std_nan_total == 0:
        print(f"  {CYAN}Both stable at this learning rate.{RESET}")
        print(f"  {DIM}Try --lr 0.5 or higher to provoke instability.{RESET}")
    else:
        print(f"  {YELLOW}Results mixed — try higher --lr or --inject flag.{RESET}")
    print(f"{BOLD}{'═' * 62}{RESET}\n")


def plot_ascii(std_hist: dict, wheel_hist: dict, epochs: int):
    """ASCII loss curve — works without matplotlib."""
    print(f"\n{BOLD}  Loss curves (ASCII){RESET}")
    print(f"  {DIM}Standard={YELLOW}─{DIM}  WheelGrad={CYAN}─{RESET}\n")

    height = 12
    width  = min(epochs, 60)
    step   = max(1, epochs // width)

    std_l   = std_hist['losses'][::step][:width]
    wheel_l = wheel_hist['losses'][::step][:width]

    all_valid = [l for l in std_l + wheel_l if not np.isnan(l)]
    if not all_valid:
        print(f"  {RED}All losses are NaN — complete training failure{RESET}\n")
        return

    lo, hi = min(all_valid), max(all_valid)
    span = max(hi - lo, 1e-6)

    def norm(v):
        if np.isnan(v): return -1
        return int((1 - (v - lo) / span) * (height - 1))

    # Build grid
    grid = [[' '] * width for _ in range(height)]
    for col, (s, w) in enumerate(zip(std_l, wheel_l)):
        rs, rw = norm(s), norm(w)
        if rs >= 0: grid[rs][col] = f'S'
        if rw >= 0: grid[rw][col] = f'W'
        if rs == rw and rs >= 0: grid[rs][col] = '*'
        if rs < 0 and col < width:
            for r in range(height): grid[r][col] = '!'  # NaN column

    for row_i, row in enumerate(grid):
        label = f"{hi - (hi - lo) * row_i / (height - 1):.3f}" if row_i % 3 == 0 else "     "
        line = ''.join(
            f"\033[91m{c}\033[0m" if c == 'S' else
            f"\033[96m{c}\033[0m" if c == 'W' else
            f"\033[93m{c}\033[0m" if c == '*' else
            f"\033[91m{c}\033[0m" if c == '!' else c
            for c in row
        )
        print(f"  {label:6}│{line}")
    print(f"  {'':6}└{'─' * width}")
    print(f"  {'':6} epoch 1{' ' * (width - 16)}epoch {epochs}")
    print(f"\n  S={YELLOW}Standard{RESET}  W={CYAN}WheelGrad{RESET}  *=overlap  !={RED}NaN{RESET}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='WheelGrad vs Standard training stability comparison'
    )
    parser.add_argument('--lr',      type=float, default=0.3,
                        help='Learning rate (default 0.3 — intentionally high)')
    parser.add_argument('--epochs',  type=int,   default=20,
                        help='Number of training epochs')
    parser.add_argument('--seed',    type=int,   default=42,
                        help='Random seed')
    parser.add_argument('--batch',   type=int,   default=32,
                        help='Batch size')
    parser.add_argument('--inject',  action='store_true',
                        help='Inject all-zero batch to trigger 0/0 scenario')
    parser.add_argument('--quiet',   action='store_true',
                        help='Suppress per-epoch output')
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}WheelGrad Stability Experiment{RESET}")
    print(f"{DIM}lr={args.lr}  epochs={args.epochs}  seed={args.seed}"
          f"  inject={'yes' if args.inject else 'no'}{RESET}\n")

    print(f"{BOLD}── Standard IEEE 754 Training ──{RESET}")
    t0 = time.time()
    std_hist = run_experiment(
        'standard', args.lr, args.epochs, args.seed,
        args.inject, args.batch, args.quiet
    )
    print(f"  {DIM}({time.time() - t0:.1f}s){RESET}\n")

    print(f"{BOLD}── WheelGrad Training ──{RESET}")
    t0 = time.time()
    wheel_hist = run_experiment(
        'wheel', args.lr, args.epochs, args.seed,
        args.inject, args.batch, args.quiet
    )
    print(f"  {DIM}({time.time() - t0:.1f}s){RESET}")

    plot_ascii(std_hist, wheel_hist, args.epochs)
    print_summary(std_hist, wheel_hist, args.lr, args.inject)


if __name__ == '__main__':
    main()

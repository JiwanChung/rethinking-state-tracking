"""Curriculum training and length-generalization evaluation (the protocol of Table 2).

1. Train on length 2. When test accuracy is >= 0.95 for 5 consecutive epochs,
   double the length (capped at 60) and draw a fresh dataset.
2. The optimizer and scheduler carry over across curriculum stages.
3. Keep the best (length, test accuracy) weights and restore them at the end.
4. If the curriculum reached 60, evaluate on lengths 100, 200, ..., 1000.
   The max-passing length is the largest one with accuracy >= 0.90; it is 60 if
   none passes and 0 if the curriculum did not reach 60.
FP32 throughout.
"""

import time

import torch
import torch.nn.functional as F

from state_tracking.data import get_state_tracking_dataloaders
from state_tracking.groups import GROUPS
from state_tracking.models import StateTrackingModel


def train_epoch(model, loader, optimizer, device, group_order):
    """One epoch with a cross-entropy loss at every position."""
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, group_order), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += (preds == labels).sum().item()
        total_tokens += labels.numel()
    return total_loss / len(loader.dataset), total_correct / total_tokens


@torch.no_grad()
def evaluate(model, loader, device, group_order):
    """Loss and accuracy over every position."""
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, group_order), labels.view(-1), reduction="sum")
        preds = logits.argmax(dim=-1)
        total_loss += loss.item()
        total_correct += (preds == labels).sum().item()
        total_tokens += labels.numel()
    return {"loss": total_loss / total_tokens, "accuracy": total_correct / total_tokens}


def run_curriculum_training(
    group_name,
    mixer,
    d_model=698,
    n_layers=1,
    d_state=64,
    dropout=0.0,
    lr=1e-3,
    weight_decay=0.01,
    scheduler_name="cosine",
    batch_size=256,
    max_epochs=500,
    start_len=2,
    max_len=60,
    curriculum_threshold=0.95,
    curriculum_patience=5,
    eval_lengths=(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
    gen_threshold=0.90,
    train_examples=10000,
    test_examples=2000,
    seed=42,
    device=None,
    verbose=True,
):
    """Train one model with the curriculum and evaluate length generalization.

    Returns a dict with the trained `model` (best weights restored),
    `max_passing`, `final_test_acc`, `final_seq_len`, `training_succeeded`,
    `curriculum_history`, `length_generalization`, `total_epochs`, `elapsed`,
    `best_state_dict` (CPU), `best_seq_len_seen`, `best_test_acc`, `n_params`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    group = GROUPS[group_name]()

    model = StateTrackingModel(
        group_order=group.order, mixer=mixer, d_model=d_model,
        n_layers=n_layers, d_state=d_state, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"[train] mixer={mixer}  d_model={d_model}  n_layers={n_layers}  "
              f"d_state={d_state}  params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5)
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10)
    elif scheduler_name == "fixed":
        scheduler = None
    else:
        raise ValueError(f"unknown scheduler_name={scheduler_name!r}")

    curr_len = start_len
    passing_streak = 0
    seed_offset = 0
    total_epochs = 0
    curriculum_history = []

    best_test_acc = -1.0
    best_curr_len = start_len
    best_state_dict = None

    train_loader, test_loader, _ = get_state_tracking_dataloaders(
        group_name, curr_len, train_examples=train_examples,
        test_examples=test_examples, batch_size=batch_size, seed=seed + seed_offset)
    if verbose:
        print(f"[train] === Training (seq_len={curr_len}) ===")

    start_time = time.time()
    for _ in range(max_epochs):
        total_epochs += 1
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, group.order)
        test_acc = evaluate(model, test_loader, device, group.order)["accuracy"]

        if scheduler_name == "cosine":
            scheduler.step()
        elif scheduler_name == "plateau":
            scheduler.step(test_acc)

        # Keep the best (curr_len, test_acc): longer lengths win, then accuracy.
        if (curr_len, test_acc) > (best_curr_len, best_test_acc):
            best_curr_len = curr_len
            best_test_acc = test_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            lr_cur = optimizer.param_groups[0]["lr"]
            print(f"  ep {total_epochs:>3} (len={curr_len:>3}) | "
                  f"train {train_loss:.4f}/{train_acc:.4f} | "
                  f"test {test_acc:.4f} | lr {lr_cur:.2e}")

        passing_streak = passing_streak + 1 if test_acc >= curriculum_threshold else 0
        if passing_streak >= curriculum_patience:
            curriculum_history.append({"seq_len": curr_len, "best_acc": best_test_acc,
                                       "epochs": total_epochs})
            if curr_len >= max_len:
                if verbose:
                    print(f"[train] curriculum reached max_len={max_len} with acc={test_acc:.4f}")
                break
            curr_len = min(curr_len * 2, max_len)
            passing_streak = 0
            seed_offset += 1
            train_loader, test_loader, _ = get_state_tracking_dataloaders(
                group_name, curr_len, train_examples=train_examples,
                test_examples=test_examples, batch_size=batch_size, seed=seed + seed_offset)
            if verbose:
                print(f"[train] === Curriculum advanced: seq_len={curr_len} ===")

    elapsed = time.time() - start_time

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    final_metrics = evaluate(model, test_loader, device, group.order)
    training_succeeded = best_curr_len >= max_len and best_test_acc >= curriculum_threshold

    if verbose:
        print(f"[train] best: len={best_curr_len}, test_acc={best_test_acc:.4f}; "
              f"succeeded={training_succeeded}  epochs={total_epochs}  time={elapsed:.1f}s")

    gen_results = {}
    max_passing_length = best_curr_len if training_succeeded else 0
    if training_succeeded:
        for eval_len in eval_lengths:
            _, eval_loader, _ = get_state_tracking_dataloaders(
                group_name, int(eval_len), train_examples=100,
                test_examples=test_examples, batch_size=batch_size, seed=seed + 9999)
            acc = evaluate(model, eval_loader, device, group.order)["accuracy"]
            gen_results[str(int(eval_len))] = round(acc, 4)
            if verbose:
                print(f"  [length {eval_len}] acc={acc:.4f}  "
                      f"[{'PASS' if acc >= gen_threshold else 'FAIL'}]")
            if acc >= gen_threshold:
                max_passing_length = int(eval_len)

    return {
        "model": model,
        "max_passing": int(max_passing_length),
        "final_test_acc": float(final_metrics["accuracy"]),
        "final_seq_len": int(curr_len),
        "training_succeeded": bool(training_succeeded),
        "curriculum_history": curriculum_history,
        "length_generalization": gen_results,
        "total_epochs": int(total_epochs),
        "elapsed": float(elapsed),
        "best_state_dict": best_state_dict,
        "best_seq_len_seen": int(best_curr_len),
        "best_test_acc": float(best_test_acc),
        "n_params": int(n_params),
    }

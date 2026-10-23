#!/usr/bin/env python3
"""Reproduce Table 2: max-passing length per model, task and depth.

Each cell is the best over a grid of 81 runs:
    d_state {32, 64, 128} x learning rate {1e-4, 5e-4, 1e-3}
    x scheduler {fixed, cosine, plateau} x seed {42, 43, 45}

Run part of the grid (results go to --out, one JSON per run; finished runs are
skipped, so the command can be restarted or split across GPUs with --shard):
    python scripts/main_table.py run --models tanh_rnn neg_mamba --tasks S3 --layers 1
    CUDA_VISIBLE_DEVICES=0 python scripts/main_table.py run --shard 0/4
    CUDA_VISIBLE_DEVICES=1 python scripts/main_table.py run --shard 1/4   # ...

Build the table from the results (a cell is printed only when all 81 runs exist):
    python scripts/main_table.py table
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state_tracking.models import LABELS, MIXERS  # noqa: E402

TASKS = ("C2", "C6", "S3")
LAYERS = (1, 2)
GRID = {
    "d_state": (32, 64, 128),
    "lr": (1e-4, 5e-4, 1e-3),
    "scheduler": ("fixed", "cosine", "plateau"),
    "seed": (42, 43, 45),
}


def run_name(mixer, task, layers, d_state, lr, scheduler, seed):
    return f"{mixer}_{task}_L{layers}_d{d_state}_lr{lr}_{scheduler}_s{seed}.json"


def grid(models, tasks, layers):
    for mixer, task, L in itertools.product(models, tasks, layers):
        for d_state, lr, sched, seed in itertools.product(*GRID.values()):
            yield dict(mixer=mixer, task=task, layers=L, d_state=d_state, lr=lr,
                       scheduler=sched, seed=seed)


def cmd_run(args):
    import torch
    from state_tracking.train import run_curriculum_training

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    jobs = [j for i, j in enumerate(grid(args.models, args.tasks, args.layers)) if i % n_shards == shard]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(jobs)} runs in shard {shard}/{n_shards} on {device}")
    for j in jobs:
        path = out / run_name(**j)
        if path.exists():
            continue
        print(f"\n=== {path.name}")
        r = run_curriculum_training(
            group_name=j["task"], mixer=j["mixer"], n_layers=j["layers"],
            d_state=j["d_state"], lr=j["lr"], scheduler_name=j["scheduler"],
            seed=j["seed"], device=device, verbose=args.verbose)
        result = {**j,
                  "max_passing_length": r["max_passing"],
                  "final_test_acc": round(r["final_test_acc"], 4),
                  "training_succeeded": r["training_succeeded"],
                  "length_generalization": r["length_generalization"],
                  "curriculum_history": r["curriculum_history"],
                  "best_seq_len_seen": r["best_seq_len_seen"],
                  "params": r["n_params"],
                  "total_epochs": r["total_epochs"],
                  "time": round(r["elapsed"], 1)}
        tmp = path.with_name(path.name + ".tmp")     # never leave a half-written result
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, path)
        print(f"max-passing length {r['max_passing']}  ({r['elapsed']:.0f}s)")


def cmd_table(args):
    out = Path(args.out)
    rows, missing = [], {}
    for mixer in args.models:
        cells = []
        for task, L in itertools.product(args.tasks, args.layers):
            runs = [out / run_name(**j) for j in grid([mixer], [task], [L])]
            have = [p for p in runs if p.exists()]
            if len(have) < len(runs):
                missing[(mixer, task, L)] = len(runs) - len(have)
                cells.append("..." if have else "")
                continue
            # Grid-best: largest max-passing length, then final test accuracy.
            best = max((json.loads(p.read_text()) for p in have),
                       key=lambda d: (d["max_passing_length"], d["final_test_acc"]))
            cells.append("✗" if best["max_passing_length"] == 0 else str(best["max_passing_length"]))
        rows.append([LABELS[mixer]] + cells)

    head = ["Model"] + [f"{t} L{L}" for t, L in itertools.product(args.tasks, args.layers)]
    width = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, width))))
    print("\n✗: the curriculum did not reach length 60.  60: no evaluation length passed.")
    if missing:
        print(f"\n{len(missing)} cells are incomplete and shown as '...' (some runs done) "
              f"or blank (none done); {sum(missing.values())} runs are missing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "table"):
        p = sub.add_parser(name)
        p.add_argument("--models", nargs="+", default=list(MIXERS), choices=MIXERS)
        p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
        p.add_argument("--layers", nargs="+", type=int, default=list(LAYERS), choices=LAYERS)
        p.add_argument("--out", default="results")
        if name == "run":
            p.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
            p.add_argument("--verbose", action="store_true", help="print every epoch")
    args = ap.parse_args()
    {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()

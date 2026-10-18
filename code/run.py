import argparse
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import data
from engine import (DEVICE, FrozenMethod, GradientMethod, PriorMethod, Session,
                    backbone_macs, pretrain, stream, summarize, target_stats)
from model import Backbone, Config

OUT = data.ROOT / "results"
FINAL_SEEDS = [101, 113, 127, 131, 137, 149, 157, 163, 173, 181]
DEV_SEEDS = [11, 23, 37]
ABLATION_SEEDS = FINAL_SEEDS[:5]
HDC_GRID = [dict(source=s, dimension=d, scale=k, lam=l) for s in ["penultimate", "state"]
            for d in [1024, 2048] for k in [1.0, 2.0, 4.0] for l in [300.0, 1000.0, 3000.0]]
LINEAR_GRID = [dict(lam=l) for l in [100.0, 300.0, 1000.0, 3000.0]]
LRS = [3e-5, 1e-4, 3e-4]
EWC_WEIGHTS = [1.0, 10.0, 100.0]


def variants(c):
    return dict(full=c, linear=replace(c, projection="linear"), mlp=replace(c, projection="mlp"),
                standard_gru=replace(c, projection="plain", refinement=False),
                no_gru=replace(c, recurrence=False), no_online_norm=replace(c, online_norm=False),
                no_smoothing=replace(c, smoothing=0.0), no_refinement=replace(c, refinement=False),
                epochs25=replace(c, epochs=25), epochs100=replace(c, epochs=100))


def base_config(task):
    rows = data.manifest(task)
    return Config(channels=rows[0]["channels"], outputs=rows[0]["outputs"])


def prepare(task, c, inner=False, prefix=data.PREFIX):
    hi = [(k, data.held_in_split(rs)) for k, rs in data.by_session(data.load(task, "held-in-calib"))]
    if inner:
        ho = [(k, data.held_out_split(rs)) for k, rs in data.by_session(data.load(task, "held-in-calib"))][-2:]
        hi = hi[:-2]
    else:
        ho = [(k, data.held_out_split(rs, prefix=prefix)) for k, rs in data.by_session(data.load(task, "held-out-calib"))]
    ystats = target_stats([p for _, p in hi])
    held_in = [Session(k, p, "train", c, ystats) for k, p in hi]
    held_out = [Session(k, p, "adapt", c, ystats) for k, p in ho]
    return held_in, held_out, ystats


_MEMO = {}


def checkpoint(task, name, c, seed, inner=False):
    key = (task, name, seed, inner)
    if key not in _MEMO:
        if len(_MEMO) > 2:
            _MEMO.clear()
        _MEMO[key] = _checkpoint(task, name, c, seed, inner)
    return _MEMO[key]


def _checkpoint(task, name, c, seed, inner=False):
    path = OUT / "checkpoints" / task / ("inner" if inner else "outer") / f"{name}_s{seed}.pt"
    held_in, held_out, ystats = prepare(task, c, inner)
    if path.exists():
        rec = torch.load(path, map_location=DEVICE, weights_only=False)
        model = Backbone(c).to(DEVICE)
        model.load_state_dict(rec["state"])
        return model, rec, held_in, held_out, ystats
    model, info = pretrain(held_in, c, seed)
    rec = dict(config=asdict(c), seed=seed, state=model.state_dict(), info=info,
               macs=backbone_macs(model), sessions=[s.name for s in held_in])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    return model, rec, held_in, held_out, ystats


def methods(sel):
    g = sel["gradient"]
    return dict(
        frozen=lambda: FrozenMethod(),
        finetune_full=lambda: GradientMethod("full", lr=g["finetune_full"]),
        finetune_readout=lambda: GradientMethod("readout", lr=g["finetune_readout"]),
        ewc=lambda: GradientMethod("full", lr=g["ewc"], ewc=g["ewc_weight"]),
        replay=lambda: GradientMethod("full", lr=g["replay"], replay=1),
        closed_linear=lambda: PriorMethod("linear", **sel["linear"]),
        hdc=lambda: PriorMethod("hdc", **sel["hdc"]),
        hdc_bound=lambda: PriorMethod("bound", **sel["hdc"]),
    )


def inner_acc(make, task, c):
    out = []
    for sd in DEV_SEEDS:
        model, rec, hi, ho, ys = checkpoint(task, "full", c, sd, inner=True)
        out.append(summarize(stream(make(), model, hi, ho, c, sd, ys))["acc"])
        torch.cuda.empty_cache()
    return float(np.mean(out))


def argbest(grid, make, task, c, kind, log):
    scores = []
    for g in grid:
        v = inner_acc(lambda: make(g), task, c)
        scores.append(v)
        log.append(dict(kind=kind, **g, acc=v))
        print(task, kind, g, round(v, 4), flush=True)
    return grid[int(np.argmax(scores))]


def select(task):
    path = OUT / "selection" / f"{task}.json"
    if path.exists():
        return json.loads(path.read_text())
    c = base_config(task)
    log = []
    hdc = argbest(HDC_GRID, lambda g: PriorMethod("hdc", **g), task, c, "hdc", log)
    linear = argbest(LINEAR_GRID, lambda g: PriorMethod("linear", **g), task, c, "closed_linear", log)
    grad = {}
    for name, mk in [("finetune_full", lambda g: GradientMethod("full", **g)),
                     ("finetune_readout", lambda g: GradientMethod("readout", **g)),
                     ("replay", lambda g: GradientMethod("full", replay=1, **g))]:
        grad[name] = argbest([dict(lr=l) for l in LRS], mk, task, c, name, log)["lr"]
    e = argbest([dict(lr=l, ewc=w) for l in LRS for w in EWC_WEIGHTS], lambda g: GradientMethod("full", **g), task, c, "ewc", log)
    grad["ewc"], grad["ewc_weight"] = e["lr"], e["ewc"]
    out = dict(hdc=hdc, linear=linear, gradient=grad, log=log, dev_seeds=DEV_SEEDS, protocol="inner: last two held-in sessions as pseudo-new sessions")
    data.write_json(path, out)
    return out


def run(task, variant, seed, method_names):
    c = variants(base_config(task))[variant]
    sel = select(task)
    model, rec, hi, ho, ys = checkpoint(task, variant, c, seed)
    for name in method_names:
        path = OUT / "stream" / task / variant / f"{name}_s{seed}.json"
        if path.exists():
            continue
        out = stream(methods(sel)[name](), model, hi, ho, c, seed, ys)
        out.update(summary=summarize(out), task=task, variant=variant, seed=seed, method=name,
                   backbone=dict(macs=rec["macs"], info=rec["info"]))
        data.write_json(path, out)


DOSES = [0.125, 0.25, 0.5]
BUDGET = ["epochs25", "epochs100"]


def dose(task, seed):
    c = base_config(task)
    sel = select(task)
    model, rec, _, _, _ = checkpoint(task, "full", c, seed)
    for name in ["hdc", "closed_linear", "finetune_full", "replay"]:
        path = OUT / "dose" / task / f"{name}_s{seed}.json"
        if path.exists():
            continue
        rows = []
        for p in DOSES:
            hi, ho, ys = prepare(task, c, prefix=p)
            for s in ho:
                m = np.zeros(len(s.score), bool)
                m[int(round(len(s.score) * 0.5)) + data.GAP:] = True
                s.score = s.score & m
            rows.append(summarize(stream(methods(sel)[name](), model, hi, ho, c, seed, ys)))
            torch.cuda.empty_cache()
        data.write_json(path, dict(task=task, method=name, seed=seed, dose=DOSES,
                                   acc_held_out=[r["acc_held_out"] for r in rows], acc=[r["acc"] for r in rows],
                                   adapt_macs=[r["adapt_macs"] for r in rows]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "main", "ablation", "budget", "dose"])
    ap.add_argument("--tasks", nargs="+", default=["h1", "m1", "m2"])
    a = ap.parse_args()
    for task in a.tasks:
        if a.stage == "select":
            select(task)
        elif a.stage == "main":
            for seed in FINAL_SEEDS:
                run(task, "full", seed, list(methods(select(task))))
        elif a.stage == "budget":
            for v in BUDGET:
                for seed in ABLATION_SEEDS:
                    run(task, v, seed, ["frozen", "hdc", "finetune_full"])
        elif a.stage == "dose":
            for seed in ABLATION_SEEDS:
                dose(task, seed)
        else:
            for v in [v for v in variants(base_config(task)) if v != "full" and v not in BUDGET]:
                for seed in ABLATION_SEEDS:
                    run(task, v, seed, ["frozen", "hdc", "finetune_full"])

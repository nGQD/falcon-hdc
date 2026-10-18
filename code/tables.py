import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TASKS = ["h1", "m1", "m2"]
MAIN = ["hdc", "closed_linear", "frozen", "finetune_full", "finetune_readout", "ewc", "replay", "hdc_bound"]
FALCON = ["wf_zeroshot", "wf_oracle", "wf_continual", "rnn_zeroshot", "rnn_oracle", "rnn_finetune"]
LABEL = dict(hdc="HDC", closed_linear="Closed form", frozen="Frozen", finetune_full="FT-full",
             finetune_readout="FT-readout", ewc="EWC", replay="Replay", hdc_bound="HDC-bound",
             wf_zeroshot="WF zero-shot", wf_oracle="WF per-session", wf_continual="WF sequential",
             rnn_zeroshot="RNN zero-shot", rnn_oracle="RNN per-session", rnn_finetune="RNN sequential")
VARIANTS = ["linear", "mlp", "standard_gru", "no_gru", "no_online_norm", "no_smoothing", "no_refinement"]


def stat(task, name):
    return json.loads((RES / "stats" / f"{task}_{name}.json").read_text())


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace("-", "−")


def sci(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(x)))
    return f"{x / 10 ** e:.1f}e{e}"


def falcon_rows(task):
    d = defaultdict(list)
    for f in glob.glob(str(RES / "stream" / task / "falcon" / "*.json")):
        j = json.loads(Path(f).read_text())
        d[j["method"]].append(j["summary"])
    return {m: {k: float(np.mean([s[k] for s in v])) for k in ("acc", "acc_held_out", "acc_held_in", "bwt", "adapt_macs")} | {"sd": float(np.std([s["acc"] for s in v], ddof=1)), "n": len(v)}
            for m, v in d.items()}


def main_table():
    lines = []
    for t in TASKS:
        agg = stat(t, "aggregate")
        lines += [f"\n### {t.upper()}\n", "| Method | ACC (95% CI) | Held-in | Held-out | BWT | Adapt MACs |", "|---|---|---|---|---|---|"]
        for m in MAIN:
            a = agg[m]
            lines.append(f"| {LABEL[m]} | {fmt(a['acc']['mean'])} ({fmt(a['acc']['ci_low'])}, {fmt(a['acc']['ci_high'])}) | "
                         f"{fmt(a['acc_held_in']['mean'])} | {fmt(a['acc_held_out']['mean'])} | {fmt(a['bwt']['mean'])} | {sci(a['adapt_macs']['mean'])} |")
        for m, a in sorted(falcon_rows(t).items(), key=lambda kv: FALCON.index(kv[0])):
            lines.append(f"| {LABEL[m]} | {fmt(a['acc'])} | {fmt(a['acc_held_in'])} | {fmt(a['acc_held_out'])} | {fmt(a['bwt'])} | {sci(a['adapt_macs'])} |")
    return "\n".join(lines)


def compact_main():
    agg = {t: stat(t, "aggregate") for t in TASKS}
    fal = {t: falcon_rows(t) for t in TASKS}
    head = "| Method | " + " | ".join(f"{t.upper()} {k}" for k in ("ACC", "held-out", "BWT") for t in TASKS) + " |"
    lines = [head, "|:--|" + ":-:|" * 9]
    for m in MAIN:
        lines.append(f"| {LABEL[m]} | " + " | ".join(fmt(agg[t][m][k]["mean"]) for k in ("acc", "acc_held_out", "bwt") for t in TASKS) + " |")
    for m in FALCON:
        lines.append(f"| {LABEL[m]} | " + " | ".join(fmt(fal[t][m][k]) for k in ("acc", "acc_held_out", "bwt") for t in TASKS) + " |")
    return "\n".join(lines)


def macs_table():
    agg = {t: stat(t, "aggregate") for t in TASKS}
    lines = ["| Method | " + " | ".join(f"{t.upper()} MACs" for t in TASKS) + " | " + " | ".join(f"{t.upper()} plasticity" for t in TASKS) + " |", "|:--|" + ":-:|" * 6]
    for m in MAIN:
        if m in ("frozen", "hdc_bound"):
            continue
        lines.append(f"| {LABEL[m]} | " + " | ".join(sci(agg[t][m]["adapt_macs"]["mean"]) for t in TASKS) + " | " + " | ".join(fmt(agg[t][m]["plasticity"]["mean"]) for t in TASKS) + " |")
    return "\n".join(lines)


def tests_table():
    lines = ["| Task | HDC vs | ACC diff | Cliff $\\delta$ | p | BWT diff | p (BWT) |", "|:--|:--|:-:|:-:|:-:|:-:|:-:|"]
    for t in TASKS:
        rows = stat(t, "paired")["rows"]
        acc = {r["other"]: r for r in rows if r["metric"] == "acc"}
        bwt = {r["other"]: r for r in rows if r["metric"] == "bwt"}
        for m in MAIN:
            if m in acc:
                a, b = acc[m], bwt[m]
                lines.append(f"| {t.upper()} | {LABEL[m]} | {fmt(a['mean_diff'])} | {fmt(a['cliff_delta'], 2)} | {a['p_holm']:.2g} | {fmt(b['mean_diff'])} | {b['p_holm']:.2g} |")
    return "\n".join(lines)


def budget_table():
    lines = ["| Task | Epochs | Frozen held-in | Frozen held-out | HDC held-out | FT-full ACC |", "|:--|:-:|:-:|:-:|:-:|:-:|"]
    for t in TASKS:
        for v, e in (("epochs25", 25), ("full", 50), ("epochs100", 100)):
            fs = {m: [json.loads(Path(f).read_text())["summary"] for f in glob.glob(str(RES / "stream" / t / v / f"{m}_s*.json"))
                      if int(Path(f).stem.split("_s")[-1]) in (101, 113, 127, 131, 137)] for m in ("frozen", "hdc", "finetune_full")}
            if not all(fs.values()):
                continue
            g = lambda m, k: float(np.mean([x[k] for x in fs[m]]))
            lines.append(f"| {t.upper()} | {e} | {fmt(g('frozen', 'acc_held_in'))} | {fmt(g('frozen', 'acc_held_out'))} | {fmt(g('hdc', 'acc_held_out'))} | {fmt(g('finetune_full', 'acc'))} |")
    return "\n".join(lines)


def tost_table():
    lines = ["| Task | Reference | Diff | 90% CI | Non-inferior at 0.03 | p |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        for k, v in stat(t, "tost").items():
            lines.append(f"| {t.upper()} | {LABEL[k]} | {fmt(v['mean_diff'])} | ({fmt(v['ci90_low'])}, {fmt(v['ci90_high'])}) | {'yes' if v['noninferior'] else 'no'} | {v['p_noninferior']:.2g} |")
    return "\n".join(lines)


def mixed_table():
    lines = ["| Task | Method | Estimate vs HDC | 95% CI | p |", "|---|---|---|---|---|"]
    for t in TASKS:
        mx = stat(t, "mixed")
        for k, v in mx.get("coefficients", {}).items():
            if k == "intercept":
                continue
            lines.append(f"| {t.upper()} | {LABEL.get(k, k)} | {fmt(v['estimate'])} | ({fmt(v['ci_low'])}, {fmt(v['ci_high'])}) | {v['p']:.2g} |")
    return "\n".join(lines)


def pareto_table():
    lines = ["| Task | Non-dominated (point estimate) | P(non-dominated): " + ", ".join(LABEL[m] for m in MAIN) + " |", "|---|---|---|"]
    for t in TASKS:
        p = stat(t, "pareto")
        lines.append(f"| {t.upper()} | {', '.join(LABEL[m] for m in p['non_dominated'])} | " + ", ".join(f"{p['prob_non_dominated'][m]:.2f}" for m in MAIN) + " |")
    return "\n".join(lines)


def ablation_table():
    lines = ["| Task | Variant | Method | Delta held-in | Delta held-out (95% CI) | Delta BWT |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        a = stat(t, "ablation")
        for v in VARIANTS:
            for m in ("hdc", "finetune_full", "frozen"):
                r = a.get(v, {}).get(m)
                if r:
                    o = r["acc_held_out"]
                    lines.append(f"| {t.upper()} | {v} | {LABEL[m]} | {fmt(r['acc_held_in']['delta'])} | {fmt(o['delta'])} ({fmt(o['ci_low'])}, {fmt(o['ci_high'])}) | {fmt(r['bwt']['delta'])} |")
    return "\n".join(lines)


def dose_table():
    lines = ["| Task | Method | " + " | ".join(f"Held-out R2 at {p}" for p in (0.125, 0.25, 0.5)) + " | Adapt MACs at 0.5 |", "|---|---|---|---|---|---|"]
    for t in TASKS:
        for m in ("hdc", "closed_linear", "finetune_full", "replay"):
            fs = glob.glob(str(RES / "dose" / t / f"{m}_s*.json"))
            if not fs:
                continue
            js = [json.loads(Path(f).read_text()) for f in fs]
            ho = np.mean([j["acc_held_out"] for j in js], 0)
            mac = np.mean([j["adapt_macs"][-1] for j in js])
            lines.append(f"| {t.upper()} | {LABEL[m]} | " + " | ".join(fmt(x) for x in ho) + f" | {sci(mac)} |")
    return "\n".join(lines)


def ratios():
    lines = []
    for t in TASKS:
        agg = stat(t, "aggregate")
        h = agg["hdc"]["adapt_macs"]["mean"]
        lines.append(f"- {t.upper()}: " + ", ".join(f"{LABEL[m]}/HDC = {agg[m]['adapt_macs']['mean'] / h:.1f}x" for m in ("finetune_full", "finetune_readout", "ewc", "replay")) +
                     f"; HDC/Closed form = {h / agg['closed_linear']['adapt_macs']['mean']:.1f}x")
    return "\n".join(lines)


if __name__ == "__main__":
    out = RES / "tables"
    out.mkdir(exist_ok=True)
    parts = dict(budget=budget_table(), compact=compact_main(), macs=macs_table(), main=main_table(), tests=tests_table(), tost=tost_table(), mixed=mixed_table(),
                 pareto=pareto_table(), ablation=ablation_table(), dose=dose_table(), ratios=ratios())
    for k, v in parts.items():
        (out / f"{k}.md").write_text(v + "\n")
    (out / "digest.md").write_text("\n\n".join(f"## {k}\n{v}" for k, v in parts.items()) + "\n")
    print(out / "digest.md")

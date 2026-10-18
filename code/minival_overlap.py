import glob
import json

import numpy as np
from pynwb import NWBHDF5IO

import data


def series(path):
    with NWBHDF5IO(path, "r") as io:
        a = io.read().acquisition["OpenLoopKinematics"]
        t = a.timestamps[:] if a.timestamps is not None else a.starting_time + np.arange(a.data.shape[0]) / a.rate
        units = io.read().units
        spikes = [np.asarray(units["spike_times"][i]) for i in range(len(units))]
        return np.asarray(a.data[:]), np.asarray(t), spikes


def counts(spikes, t):
    # H1 kinematic timestamps are in milliseconds, spike times in seconds
    t = t / 1000.0
    edges = np.append(t, t[-1] + np.median(np.diff(t)))
    return np.stack([np.histogram(s, edges)[0] for s in spikes], 1)


def main():
    rows = []
    for f in sorted(glob.glob(str(data.DATA / "000954" / "sub-HumanPitt-held-in-minival" / "*.nwb"))):
        x1, t1, s1 = series(f)
        x2, t2, s2 = series(f.replace("minival", "calib"))
        i = np.minimum(np.searchsorted(t2, t1), len(t2) - 1)
        same = (t2[i] == t1) & np.all(np.isclose(x1, x2[i]), 1)
        # spike counts of every unit, binned on the minival bin edges, compared over the span
        # in which the minival file contains spike times
        end = max(x.max() for x in s1 if len(x)) * 1000.0
        span = t1 <= end
        equal = np.all(counts(s1, t1) == counts(s2, t1), 1) if len(s1) == len(s2) else np.zeros(len(t1), bool)
        rows.append(dict(file=f.split("/")[-1], bins=int(len(t1)), duplicated=int(same.sum()),
                         bins_with_spikes=int(span.sum()), spike_counts_identical=int((equal & span).sum())))
    out = dict(files=rows, bins=sum(r["bins"] for r in rows), duplicated=sum(r["duplicated"] for r in rows),
               bins_with_spikes=sum(r["bins_with_spikes"] for r in rows),
               spike_counts_identical=sum(r["spike_counts_identical"] for r in rows))
    data.write_json(data.ROOT / "results" / "provenance" / "minival_overlap.json", out)
    print(out["bins"], out["duplicated"])


if __name__ == "__main__":
    main()

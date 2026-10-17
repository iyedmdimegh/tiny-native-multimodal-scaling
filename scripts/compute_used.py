"""Add up the GPU time and FLOPs of every full-preset run. Writes results/compute.json.

Runs are grouped by their folder under RUNS_DIR/full: grid runs sit at the top,
the LR sweep, LR check, controls, mixtures and fusion runs in subfolders.
Wall time is training only. Evaluation and data preparation are not counted.
"""

import json

from tnms.config import REPO_ROOT, RUNS_DIR


def main():
    root = RUNS_DIR / "full"
    groups = {}
    for path in sorted(root.rglob("summary.json")):
        s = json.loads(path.read_text())
        rel = path.parent.relative_to(root)
        group = rel.parts[0] if len(rel.parts) > 1 else "grid"
        g = groups.setdefault(group, {"runs": 0, "gpu_hours": 0.0, "flops_6ND": 0.0})
        g["runs"] += 1
        g["gpu_hours"] += s["wall_time_s"] / 3600
        g["flops_6ND"] += s["flops_6ND"]
    devices = {json.loads(p.read_text())["device"] for p in root.rglob("summary.json")}
    total = {k: sum(g[k] for g in groups.values()) for k in ("runs", "gpu_hours", "flops_6ND")}
    out = {"devices": sorted(devices), "total": total, "by_group": groups}
    (REPO_ROOT / "results" / "compute.json").write_text(json.dumps(out, indent=2))
    for name, g in {**groups, "total": total}.items():
        print(f"{name:10s} {g['runs']:3d} runs  {g['gpu_hours']:5.2f} GPU h  {g['flops_6ND']:.2e} FLOPs")


if __name__ == "__main__":
    main()

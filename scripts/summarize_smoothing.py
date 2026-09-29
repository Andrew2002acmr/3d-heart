"""Export compact JSON summaries and complete Markdown metrics from an existing run."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def summarize(source, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new empty summary directory")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not data.get("completed"):
        raise ValueError("Experiment is incomplete")
    out.mkdir(parents=True, exist_ok=True)
    entries = data["structures"]
    keys = list(entries[0]["variants"])
    samples = entries[0]["variants"][keys[0]]["deviation_from_baseline"]["sample_count_per_direction"]
    def stats(values):
        valid = [v for v in values if v is not None]
        return {"n": len(valid), "min": min(valid), "median": float(np.median(valid)), "max": max(valid)} if valid else {"n": 0}
    summary = {"structures": len(entries), "variants": len(entries)*len(keys),
               "all_source_hashes_unchanged": data["all_source_hashes_unchanged"],
               "missing_labels": data["missing_labels"], "profiles": {}, "defects": []}
    for key in keys:
        metrics = [e["variants"][key] for e in entries]
        summary["profiles"][key] = {
            "area_change_percent": stats([m["area_change_percent"] for m in metrics]),
            "volume_change_percent": stats([m["volume_change_percent"] for m in metrics]),
            "mean_distance": stats([m["deviation_from_baseline"]["symmetric_mean"] for m in metrics]),
            "sampled_max_distance": stats([m["deviation_from_baseline"]["sampled_max"] for m in metrics]),
            "hausdorff_upper_bound": stats([m["deviation_from_baseline"]["hausdorff_upper_bound"] for m in metrics]),
            "assessment": dict(Counter(m["assessment"]["status"] for m in metrics))}
        for e, m in zip(entries, metrics):
            changes = {k: [e["baseline"][k], m[k]] for k in
                       ("boundary_edges", "non_manifold_edges", "zero_area_triangles", "inconsistent_winding_edges", "surface_components_vertex_connected")
                       if e["baseline"][k] != m[k]}
            if changes:
                summary["defects"].append({"case": e["case"], "structure": e["name"], "profile": key, "changes": changes})
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    lines = ["# Полные метрики эксперимента smoothing", "",
             "Единицы: u — текущие координаты, не подтверждённые mm; площадь u², объём u³. "
             "Объём условный при пройденных проверках рёбер/обхода/вырожденности; самопересечения не проверены. "
             "`—` означает недостоверный/неопределённый объём. C — компоненты по общим вершинам; "
             "B — boundary edges; NM — non-manifold edges; D — треугольники с площадью ≤ 1e-12 u². "
             f"Расстояния оценены по {samples} точек в каждом направлении; max выборочный, H↑ — верхняя граница Хаусдорфа по соответствующим вершинам.", ""]
    def fmt(v, digits=3):
        return "—" if v is None else f"{v:.{digits}f}"
    for e in entries:
        lines += [f"## {e['case']} · {e['name']}", "",
            "| Вариант | V | F | Площадь | ΔA % | Объём | ΔV % | C | B | NM | D | mean u | max u | H↑ u | Отбор |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, m in [("baseline", e["baseline"]), *e["variants"].items()]:
            d = m.get("deviation_from_baseline", {})
            fields = [name, str(m["vertices"]), str(m["triangles"]), fmt(m["surface_area"]), fmt(m.get("area_change_percent", 0)),
                      fmt(m["enclosed_volume"]), fmt(m.get("volume_change_percent", 0 if m["enclosed_volume"] is not None else None)),
                      *[str(m[k]) for k in ("surface_components_vertex_connected", "boundary_edges", "non_manifold_edges", "zero_area_triangles")],
                      fmt(d.get("symmetric_mean", 0)), fmt(d.get("sampled_max", 0)), fmt(d.get("hausdorff_upper_bound", 0)),
                      m.get("assessment", {}).get("status", "baseline")]
            lines.append("| " + " | ".join(fields) + " |")
        lines.append("")
    (out / "metrics_table.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.metrics, args.out), indent=2))

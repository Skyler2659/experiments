"""Report completeness and paired outcomes without promoting missing runs to wins."""
from collections import Counter, defaultdict
import csv
from .campaign import write


def summarize(campaign):
    rows = campaign.rows()
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["task"])].append(row)
    tasks = []
    for (strategy, task), group in grouped.items():
        scored = [r for r in group if r["status"] == "SCORED" and not r["result"].get("probe_only")]
        failed = [r for r in group if r["status"] == "METHOD_FAILURE"]
        wins = sum(r["result"]["official_success"] for r in scored)
        complete = len(scored) + len(failed) == len(group)
        tasks.append({"strategy": strategy, "task": task, "dimension": group[0]["dimension"],
                      "planned": len(group), "scored": len(scored), "method_failures": len(failed), "successes": wins,
                      "unresolved": len(group) - len(scored) - len(failed),
                      "complete": complete, "success_rate": wins / len(group) if complete else None,
                      "completed_success_rate": wins / (len(scored) + len(failed)) if scored or failed else None,
                      "native_mean_score": sum(r["result"]["score_0_100"] for r in scored) / len(scored) if scored else None})
    paired = defaultdict(dict)
    for row in rows:
        if row["status"] in ("SCORED", "METHOD_FAILURE") and not (row["result"] or {}).get("probe_only"):
            paired[(row["task"], row["sample"], row["native_task"], row["layout_id"])][row["strategy"]] = row
    required = set(campaign.manifest["spec"].get("strategies", [j["strategy"] for j in campaign.manifest["jobs"]]))
    paired_count = sum(set(v) == required for v in paired.values())
    summary = {"campaign": campaign.manifest["name"], "purpose":campaign.manifest['spec'].get('purpose','experiment'),
               "planned_runs": len(rows), "states": dict(Counter(r["status"] for r in rows)),
               "complete": all(r["complete"] for r in tasks), "paired_samples_complete": paired_count,
               "aggregation": "task subset; not the official full 42-task leaderboard average", "tasks": tasks}
    write(campaign.root / "summary.json", summary)
    with (campaign.root / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(tasks[0]))
        writer.writeheader(); writer.writerows(tasks)
    return summary

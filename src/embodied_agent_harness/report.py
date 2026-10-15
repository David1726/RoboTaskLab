"""Standalone local result browser; no network service or external assets required."""

import argparse
import json
import math
import random
from pathlib import Path

from .metrics import force_summary
from .spec import read_json, write_json


def display_values(value):
    """Undefined legacy diagnostics are null for strict browser JSON; preserve raw evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: display_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [display_values(v) for v in value]
    return value


def clustered_interval(pairs, samples=2000):
    groups = {}
    for pair in pairs:
        groups.setdefault(pair["task_id"], []).append(pair["success_delta"])
    means = [sum(v) / len(v) for v in groups.values()]
    if not means:
        raise ValueError("nonempty task groups required")
    rng = random.Random(0)
    values = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(samples))
    status = (
        "insufficient_tasks"
        if len(groups) < 5
        else "degenerate_resampling"
        if values[int(samples * 0.025)] == values[int(samples * 0.975)]
        else "exploratory"
    )
    return {
        "lower": values[int(samples * 0.025)] if status == "exploratory" else None,
        "upper": values[int(samples * 0.975)] if status == "exploratory" else None,
        "status": status,
        "task_clusters": len(groups),
        "estimate": sum(means) / len(means),
        "method": "task-cluster bootstrap, equal task weight",
        "warning": "Few task clusters yield an unstable interval; not publication evidence.",
    }


def render_transfer_report(root: Path):
    report = read_json(root / "report.json")
    physical = report.get("evaluation_track") == "physical_icl"
    rows_by_condition = {
        name: read_json(root / name / "summary.json")["rows"] for name in report["conditions"]
    }
    from .transfer import paired_comparison

    report["conditions"] = paired_comparison(rows_by_condition)["conditions"]
    if physical:
        from .physical_icl import physical_summary

        report["physical_icl"] = physical_summary(rows_by_condition)
    for condition, result in report["conditions"].items():
        condition_rows = read_json(root / condition / "summary.json")["rows"]
        diagnostic = force_summary(condition_rows)
        if physical:
            report["physical_icl"][condition].update(diagnostic)
        result["paired_success_delta_ci95"] = (
            report["physical_icl"][condition]["paired_macro_delta_ci95"]
            if physical
            else clustered_interval(result["pairs"])
        )
        result["physical_icl_score"] = (
            report["physical_icl"][condition]["world_family_macro_success"] if physical else None
        )
        result["force_qualified_success_rate"] = diagnostic["force_qualified_success_rate"]
        result["max_sampled_peak_wrist_force_N"] = diagnostic["max_sampled_peak_wrist_force_N"]
    write_json(root / "report.json", report)
    episodes = {}
    for condition in report["conditions"]:
        summary = read_json(root / condition / "summary.json")
        episodes[condition] = [
            {
                "id": row["case"]["id"],
                "instruction": row["instruction"],
                "task_id": row["case"]["task_id"],
                "success": row["success"],
                "steps": row["steps"],
                "calls": row["skill_calls"],
                "termination": row["termination"],
                "error": row["error"],
                "peak_force": row.get("peak_wrist_force_N"),
                "quality": row.get("force_qualified_success")
                if row.get("force_metric_status") == "task_declared"
                else None,
                "metric_version": row.get("metric_version", "legacy v1"),
                "world": row["case"].get("world_id"),
                "family": row["case"].get("task_family"),
                "evaluation": row.get("evaluation_metrics", {}),
                "video": f"{condition}/{row['case']['id']}/rollout.mp4",
                "result": f"{condition}/{row['case']['id']}/result.json",
            }
            for row in summary["rows"]
        ]
    payload = json.dumps(
        display_values({"report": report, "episodes": episodes}),
        ensure_ascii=False,
        allow_nan=False,
    ).replace("<", "\\u003c")
    content = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>REPORT_TITLE</title>
<style>
body{font:16px system-ui;margin:32px auto;max-width:1100px;padding:0 20px;
color:#20303c;background:#fafafa}
h1{font-size:28px}p{line-height:1.6}select{font:inherit;padding:8px}
table{border-collapse:collapse;width:100%;margin:20px 0}
th,td{text-align:left;border-bottom:1px solid #ddd;padding:10px}
video{width:100%;max-height:550px;background:#151c22}
.table{overflow-x:auto}footer{margin-top:32px;color:#59666e}
</style>
<h1>REPORT_TITLE</h1>
<p>Frozen agent and skills; disjoint source and target tasks.
Observations include privileged state. Policy identity is recorded in the manifest.</p>
<div class="table"><table><thead><tr><th>Condition</th><th>Success</th>
<th>Mean steps (all)</th><th>Mean calls</th><th>Paired success Δ</th>
<th>MACRO_LABEL Δ / CI95</th><th>World/family macro SR</th><th>Task force constraint</th>
<th>Max sampled wrist load (N)</th>
</tr></thead><tbody id="metrics"></tbody></table></div>
<p>Low step counts on failed episodes do not indicate better efficiency.
With few CLUSTER_LABEL clusters, confidence intervals are unstable.</p>
<p>Raw wrist load is sampled at 20 Hz and includes load/dynamic effects.
Force constraints are optional task specifications, separate from completion.</p>
<p id="source-cost"></p>
<label>Condition <select id="condition"></select></label>
<label>Episode <select id="episode"></select></label>
<p id="description" aria-live="polite"></p><video id="video" controls></video>
<pre id="evaluation"></pre>
<p><a id="result">Episode JSON</a> · <a id="context">Input context</a> ·
<a href="report.json">Metrics JSON</a> · <a href="experience-bank.json">Source bank</a></p>
<footer>Episode inputs, manifests and trajectories retain execution provenance.
Reports include failed episodes and implementation errors.</footer>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const sel=document.getElementById('condition'), ep=document.getElementById('episode');
const pp=x=>(100*x).toFixed(1)+' pp';
if(data.report.source_cost){
 const c=data.report.source_cost;
 document.getElementById('source-cost').textContent=
 `Source collection: ${c.episodes} episodes, ${c.steps} steps, ${c.skill_calls} skill calls.`;
}
for(const [name,m] of Object.entries(data.report.conditions)){
 const o=document.createElement('option');o.value=name;o.textContent=name;sel.append(o);
 const ci=m.paired_success_delta_ci95;
 const values=[name,`${m.successes}/${m.episodes}`,m.mean_steps.toFixed(1),
 m.mean_skill_calls.toFixed(1),pp(m.paired_success_delta),
 ci.lower===null || ci.upper===null ?
 `${pp(ci.estimate)}; CI not reported (${ci.world_clusters ?? ci.task_clusters} clusters)` :
 `${pp(ci.estimate)} [${pp(ci.lower)}, ${pp(ci.upper)}]`,
 m.physical_icl_score===null ? '—' : (100*m.physical_icl_score).toFixed(1)+'%',
 m.force_qualified_success_rate===null ? '—' :
 (100*m.force_qualified_success_rate).toFixed(1)+'%',
 m.max_sampled_peak_wrist_force_N===null ? '—' : m.max_sampled_peak_wrist_force_N.toFixed(1)];
 const tr=document.createElement('tr');
 for(const value of values){
  const td=document.createElement('td');td.textContent=value;tr.append(td)
 }
 document.getElementById('metrics').append(tr)
}
function show(){
 const r=data.episodes[sel.value][Number(ep.value)];
 document.getElementById('description').textContent=
 `${r.instruction} | task ${r.task_id} | success: ${r.success}`+
 ` | ${r.steps} steps | ${r.calls} calls | ${r.termination}`+
 (r.error ? ` | error: ${r.error}` : '');
 document.getElementById('video').src=r.video;
 document.getElementById('evaluation').textContent=JSON.stringify({world:r.world,
 family:r.family,metric_version:r.metric_version,peak_wrist_force_N:r.peak_force,
 task_force_constraint_success:r.quality,
 independent_evaluation:r.evaluation},null,2);
 document.getElementById('result').href=r.result;
 document.getElementById('context').href=`${sel.value}/${r.id}/context.json`;
}
function choose(){
 ep.replaceChildren();
 data.episodes[sel.value].forEach((r,i)=>{
  const o=document.createElement('option');o.value=i;o.textContent=r.id;ep.append(o)
 });
 show();
}
sel.addEventListener('change',choose);ep.addEventListener('change',show);choose();
</script></html>"""
    title = "Physical ICL: cross-family generalization" if physical else "Agent context transfer"
    label = "World/family macro" if physical else "Task macro"
    content = content.replace("REPORT_TITLE", title).replace("MACRO_LABEL", label)
    content = content.replace("CLUSTER_LABEL", "physical world" if physical else "task")
    (root / "report.html").write_text(content.replace("PAYLOAD", payload), encoding="utf-8")
    return root / "report.html"


def render_benchmark_report(root: Path):
    """Task completion report; never interpret it as an ICL context comparison."""
    summary = read_json(root / "summary.json")
    manifest = read_json(root / "manifest.json")
    source = (
        manifest["panel"].get("evaluation_track") == "physical_icl"
        and manifest["split"] == "development"
    )
    title = (
        "Physical interaction source collection" if source else "Agent benchmark: task execution"
    )
    episodes = [
        {
            "id": r["case"]["id"],
            "task": r["case"]["task_id"],
            "instruction": r["instruction"],
            "success": r["success"],
            "steps": r["steps"],
            "termination": r["termination"],
            "error": r["error"],
            "peak_wrist_force_N": r.get("peak_wrist_force_N"),
            "task_force_constraint_success": r.get("force_qualified_success")
            if r.get("force_metric_status") == "task_declared"
            else None,
            "metric_version": r.get("metric_version", "legacy v1"),
            "evaluation_metrics": r.get("evaluation_metrics", {}),
        }
        for r in summary["rows"]
    ]
    payload = json.dumps(display_values(episodes), ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TITLE</title>
<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:0 18px;
background:#fafafa;color:#20303c}p{line-height:1.6}video{width:100%;max-height:540px}
select{font:inherit;padding:8px}pre{white-space:pre-wrap}</style>
<h1>TITLE</h1><p>SCORE</p>
<p>Task completion with state observations and participant-supplied policy code.
Context comparisons use separate paired transfer reports.</p>
<p>Raw wrist load is diagnostic unless a task limit and rationale are declared.</p>
<label>Episode <select id="episode"></select></label><pre id="description"></pre>
<video id="video" controls></video><p><a id="result">Episode JSON</a> ·
<a href="summary.json">Summary</a> · <a href="manifest.json">Protocol</a></p>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const select=document.getElementById('episode');
data.forEach((r,i)=>{const o=document.createElement('option');
o.value=i;o.textContent=r.id;select.append(o)});
function show(){const r=data[Number(select.value)];
document.getElementById('description').textContent=JSON.stringify(r,null,2);
document.getElementById('video').src=`${r.id}/rollout.mp4`;
document.getElementById('result').href=`${r.id}/result.json`;}
select.addEventListener('change',show);show();</script></html>"""
    page = page.replace("TITLE", title).replace(
        "SCORE", f"Task success: {summary['successes']}/{summary['episodes']}."
    )
    page = page.replace("PAYLOAD", payload)
    (root / "report.html").write_text(page, encoding="utf-8")
    return root / "report.html"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    print(
        render_transfer_report(root)
        if (root / "report.json").exists()
        else render_benchmark_report(root)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

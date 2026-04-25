import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

# Figure 1: Coupling cliff
with open(REPO / "tier1_coupling_results.json") as f:
    coupling = json.load(f)
results = coupling["results"]
wer_levels = sorted([int(k.split("_")[1]) for k in results if k.startswith("wer_")])
mean_q = [results[f"wer_{w}"]["mean_quality"] for w in wer_levels]
std_q  = [results[f"wer_{w}"]["std_quality"]  for w in wer_levels]

plt.figure(figsize=(8, 4.5), dpi=130)
plt.errorbar(wer_levels, mean_q, yerr=std_q,
             marker="o", linewidth=2, markersize=7, capsize=4,
             color="#d1495b", ecolor="#d1495b80",
             label=f"{coupling['model']} (n={coupling['n_queries_per_wer']} per WER)")
plt.axvspan(0, 2, alpha=0.08, color="#2e933c", label="clean regime (WER ≤ 2%)")
plt.axvspan(2, 20, alpha=0.05, color="#d1495b", label="coupling cliff (WER > 2%)")
plt.xlabel("Injected ASR word error rate (%)")
plt.ylabel("Mean downstream LLM quality (0-1)")
plt.title("Coupling cliff: downstream LLM quality vs upstream ASR WER")
plt.grid(True, alpha=0.25, linestyle="--")
plt.ylim(0, 1.05); plt.xlim(-0.5, 20.5)
plt.legend(loc="lower left", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "coupling_cliff.png"); plt.close()
print("wrote", FIG/"coupling_cliff.png")

# Figure 2: LLM latency by context length
with open(REPO / "tier1_llm_latency_results.json") as f:
    lat = json.load(f)
rows = []
for k, v in lat.items():
    if "total_latency_ms" not in v: continue
    rows.append({
        "label": v.get("context_description", k),
        "model": v.get("model", k),
        "mean": v["total_latency_ms"]["mean"],
        "p95":  v["total_latency_ms"]["p95"],
        "std":  v["total_latency_ms"]["std"],
        "np":   v.get("num_predict", 0),
    })
rows.sort(key=lambda r: (r["model"], r["np"] or 0))
labels = [r["label"] for r in rows]
means  = [r["mean"]  for r in rows]
p95s   = [r["p95"]   for r in rows]
stds   = [r["std"]   for r in rows]
x = np.arange(len(labels)); width = 0.38

plt.figure(figsize=(9, 4.8), dpi=130)
plt.bar(x - width/2, means, width, yerr=stds, capsize=3,
        color="#2e4a7b", alpha=0.85, label="mean ± std")
plt.bar(x + width/2, p95s, width, color="#d1495b", alpha=0.85, label="P95")
plt.xticks(x, labels, rotation=18, ha="right", fontsize=9)
plt.ylabel("End-to-end LLM latency (ms)")
plt.title(f"LLM latency by context length - {rows[0]['model']}")
plt.grid(True, alpha=0.25, linestyle="--", axis="y")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "llm_latency.png"); plt.close()
print("wrote", FIG/"llm_latency.png")

# Figure 3: Social preview 1280x640
fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor("#0d1117")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1280); ax.set_ylim(0, 640); ax.set_axis_off()
ax.set_facecolor("#0d1117")
ax.text(64, 560, "PAVO-Bench", fontsize=58, color="#ffffff", fontweight="bold", family="DejaVu Sans")
ax.text(64, 510, "Pipeline-Aware Voice Orchestration · TMLR 2026", fontsize=22, color="#8b949e", family="DejaVu Sans")
def stat(xp, big, small, color):
    ax.text(xp, 320, big,   fontsize=72, color=color, fontweight="bold", family="DejaVu Sans")
    ax.text(xp, 270, small, fontsize=18, color="#c9d1d9", family="DejaVu Sans")
stat(64,  "-10.3%",  "P95 latency, H100 (p = 2e-6)",                  "#58a6ff")
stat(464, "-71%",  "energy per turn",                               "#3fb950")
stat(864, "85 K",  "meta-controller params  ·  106 s training",     "#d29922")
ax.text(64, 80, "50,000-turn benchmark  ·  A100 / H100 + Apple M3  ·  code + paper + dataset",
        fontsize=17, color="#8b949e", family="DejaVu Sans")
ax.text(64, 44, "github.com/vnmoorthy/pavo-bench",
        fontsize=17, color="#58a6ff", family="DejaVu Sans")
plt.savefig(FIG / "social-preview.png", facecolor=fig.get_facecolor(), edgecolor="none")
plt.close()
print("wrote", FIG/"social-preview.png")

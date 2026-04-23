#!/usr/bin/env python3
"""
Generate LaTeX ablation table and interpretation paragraph from ablation_results.json.
"""
import json

with open("ablation_results.json") as f:
    data = json.load(f)

oracle_cost = data["oracle_cost"]
models = data["models"]

def fmt_train(s):
    if s < 60: return f"{s:.0f}s"
    return f"{s/60:.1f}min"

ORDER = ["Logistic Reg.", "Random Forest", "XGBoost", "MLP (CE)", "MLP (PPO)"]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\caption{Routing method comparison: RL vs.\ supervised learning. Cost gap is percentage above oracle minimum.}")
lines.append(r"\label{tab:ablation_supervised}")
lines.append(r"\centering")
lines.append(r"\begin{tabular}{lccccc}")
lines.append(r"\toprule")
lines.append(r"Method & Acc (\%) & Top-3 (\%) & Cost gap (\%) & Decision ($\mu$s) & Train time \\")
lines.append(r"\midrule")
lines.append(r"Oracle & 100.0 & 100.0 & 0.00 & -- & -- \\")

for name in ORDER:
    m = models.get(name)
    if m is None:
        lines.append(f"{name} & \\multicolumn{{5}}{{c}}{{not available}} \\\\")
        continue
    lines.append(
        f"{name} & {m['acc']*100:.1f} & {m['top3']*100:.1f} & "
        f"{m['cost_gap']:.2f} & {m['lat_us']:.1f} & {fmt_train(m['train_s'])} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

table = "\n".join(lines)
print("\nLaTeX Table:")
print(table)

# Interpretation paragraph
ppo = models.get("MLP (PPO)")
best_sup_gap = min(
    (m["cost_gap"] for k, m in models.items() if m and k != "MLP (PPO)"),
    default=None
)
if ppo and best_sup_gap is not None:
    diff = best_sup_gap - ppo["cost_gap"]
    if diff > 5.0:
        para = (
            f"PPO training provides {diff:.1f}\\% lower routing cost than the best supervised "
            f"baseline ({best_sup_gap:.2f}\\% vs.\\ {ppo['cost_gap']:.2f}\\% above oracle), "
            f"justifying the RL formulation for the PAVO routing policy."
        )
    else:
        para = (
            f"Supervised classifiers achieve routing cost within {abs(diff):.1f}\\% of PPO "
            f"({best_sup_gap:.2f}\\% vs.\\ {ppo['cost_gap']:.2f}\\% above oracle), "
            f"suggesting the primary contribution of the PAVO framework is the demand vector "
            f"formulation and coupling constraints rather than the training algorithm itself."
        )
    print("\nInterpretation paragraph:")
    print(para)

# Save
with open("ablation_table.tex", "w") as f:
    f.write(table + "\n")
if ppo and best_sup_gap is not None:
    with open("ablation_paragraph.txt", "w") as f:
        f.write(para + "\n")
print("\nSaved ablation_table.tex and ablation_paragraph.txt")

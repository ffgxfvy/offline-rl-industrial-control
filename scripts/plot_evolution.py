"""plot_evolution.py — 用 results/ 下的 evaluation_results.csv 生成演进对比图。

用法：  python scripts/plot_evolution.py
输出：  <outdir>/evolution.png  <outdir>/score_distribution.png（默认 assets/）
"""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ap = argparse.ArgumentParser(description="生成三代策略对比图")
_ap.add_argument("--outdir", default=os.path.join(ROOT, "assets"),
                 help="图片输出目录（默认 <repo>/assets）")
OUTDIR = _ap.parse_args().outdir
os.makedirs(OUTDIR, exist_ok=True)
GENS = [
    ("gen2_td",      "v2 帧堆叠+时序差分特征", "#9ecae1"),
    ("gen3_history", "v3 帧堆叠历史观测",      "#6baed6"),
    ("gen4_tcn",     "v4 TCN 时序卷积网络",    "#08519c"),
]

data = {}
for key, label, color in GENS:
    p = os.path.join(ROOT, "results", key, "evaluation_results.csv")
    data[key] = pd.read_csv(p)

# ---------- 图 1：均值 + 误差棒 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

labels = [g[1] for g in GENS]
means  = [data[g[0]].score.mean() for g in GENS]
stds   = [data[g[0]].score.std()  for g in GENS]
colors = [g[2] for g in GENS]

ax = axes[0]
bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, edgecolor="#333", linewidth=0.8)
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width()/2, m + s + 0.8, f"{m:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("评测得分 (score)")
ax.set_title("三代策略得分对比（100 个随机种子的均值 ± 标准差）", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(means) + max(stds) + 6)
plt.setp(ax.get_xticklabels(), fontsize=9)

# 状态误差
err = [data[g[0]].avg_state_error.mean() for g in GENS]
ax = axes[1]
bars = ax.bar(labels, err, color=colors, edgecolor="#333", linewidth=0.8)
for b, e in zip(bars, err):
    ax.text(b.get_x() + b.get_width()/2, e + 0.02, f"{e:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("平均状态误差 (avg_state_error)")
ax.set_title("控制精度对比（越低越好）", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(err) * 1.25)
plt.setp(ax.get_xticklabels(), fontsize=9)

fig.tight_layout()
out1 = os.path.join(OUTDIR, "evolution.png")
fig.savefig(out1); plt.close(fig)
print("saved", out1)

# ---------- 图 2：100 seeds 分布箱线图 ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
box = ax.boxplot([data[g[0]].score.values for g in GENS], patch_artist=True,
                 labels=labels, widths=0.5, showmeans=True)
for patch, c in zip(box["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.85)
for i, g in enumerate(GENS, start=1):
    y = data[g[0]].score.values
    ax.scatter(np.full(len(y), i) + np.random.uniform(-0.13, 0.13, len(y)), y,
               s=6, color="#222", alpha=0.25, zorder=3)
ax.set_ylabel("评测得分 (score)")
ax.set_title("100 个随机种子下的得分分布", fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), fontsize=9)
fig.tight_layout()
out2 = os.path.join(OUTDIR, "score_distribution.png")
fig.savefig(out2); plt.close(fig)
print("saved", out2)

# ---------- 打印汇总 ----------
print("\n方法\t\t\t均值\t最大\t最小\t标准差\t累计奖励均值\t状态误差")
for key, label, _ in GENS:
    d = data[key]
    print(f"{label}\t{d.score.mean():.3f}\t{d.score.max():.3f}\t{d.score.min():.3f}\t{d.score.std():.2f}\t{d.total_reward.mean():.1f}\t{d.avg_state_error.mean():.4f}")

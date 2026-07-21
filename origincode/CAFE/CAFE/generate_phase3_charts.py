"""
阶段三: 生成综合对比图表 (Acc + Loss)
读取所有 output_optimize/*/training_metrics.json，生成:
  1. 优化路径总览 (折线图: Acc+Loss)
  2. 各层对比 (分组柱状图)
  3. 最终验证 3-seed 对比
  4. 最优实验 Loss 曲线
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── 配置 ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parent / "KMU-FED" / "output_optimize"
OUT = Path(__file__).resolve().parents[3] / "experiments" / "results" / "阶段三" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = ['#2c7bb6', '#fdae61', '#d7191c', '#1b9e77', '#7570b3',
          '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']

def load_metrics(exp_name):
    path = BASE / exp_name / "training_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def get_best_val_acc(exp_name):
    m = load_metrics(exp_name)
    if m:
        return m['mean_accuracy'] * 100
    return None

def get_fold_data(exp_name):
    """返回所有 fold 的 epoch 数据"""
    m = load_metrics(exp_name)
    if m and 'fold_histories' in m:
        return m
    return None

# ── 1. 优化路径总览 ─────────────────────────────────
def chart_optimization_path():
    """折线图: 每个优化步骤的 Acc↑ + Std 误差条"""
    steps = [
        ("baseline\nRN18+MSCeleb", "baseline"),
        ("+ImageNet\n预训练", "bb-rn18-imagenet"),
        ("+ResNet-34", "bb-rn34-msceleb"),
        ("+ResNet-50", "bb-rn50-msceleb"),
        ("+Cosine\n调度器", "sch-cos"),
        ("+Geometric\n增强", "aug-geo"),
        ("+div=3\n损失权重", "loss-div3"),
        ("+epochs=80", "cfg-ep80"),
    ]
    labels = [s[0] for s in steps]
    exps = [s[1] for s in steps]
    accs = []
    stds = []
    for e in exps:
        m = load_metrics(e)
        if m:
            accs.append(m['mean_accuracy'] * 100)
            stds.append(m['std_accuracy'] * 100)
        else:
            accs.append(None)
            stds.append(None)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(steps))
    accs_arr = np.array(accs)
    stds_arr = np.array(stds)

    ax.fill_between(x, accs_arr - stds_arr, accs_arr + stds_arr,
                     alpha=0.15, color=COLORS[0])
    ax.plot(x, accs_arr, 'o-', color=COLORS[0], linewidth=2.5, markersize=8,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLORS[0])
    for i, (a, s) in enumerate(zip(accs_arr, stds_arr)):
        ax.annotate(f'{a:.1f}%', (i, a), textcoords="offset points",
                    xytext=(0, 14), ha='center', fontsize=8, fontweight='bold',
                    color=COLORS[2])
        ax.annotate(f'±{s:.1f}%', (i, a-s), textcoords="offset points",
                    xytext=(0, -12), ha='center', fontsize=7, color='gray', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('CAFE 参数优化路径', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(60, 100)

    # 标注提升幅度
    first = accs_arr[0]
    last = accs_arr[-1]
    ax.annotate(f'+{last-first:.1f}%', xy=(len(steps)-0.3, last+2),
                fontsize=12, fontweight='bold', color=COLORS[2],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaea', alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT / "optimization-path.png")
    plt.close(fig)
    print("  OK optimization-path.png")

# ── 2. 各层对比柱状图 ─────────────────────────────────
def chart_layer_bars():
    """每层独立柱状图"""
    layers = [
        ("L1: Backbone + 预训练", [
            ("RN18\nMSCeleb", "baseline"),
            ("RN18\nImageNet", "bb-rn18-imagenet"),
            ("RN34\nImageNet", "bb-rn34-msceleb"),
            ("RN50\nImageNet", "bb-rn50-msceleb"),
        ]),
        ("L2: 优化器 + 调度器", [
            ("Adam+Exp", "bb-rn50-msceleb"),
            ("AdamW+Exp", "opt-adamw"),
            ("Adam+Cos", "sch-cos"),
            ("Adam+Plat", "sch-plateau"),
        ]),
        ("L3: 学习率 + 权重衰减", [
            ("lr=5e-4", "lr-5e4"),
            ("lr=2e-4 OK", "sch-cos"),
            ("lr=1e-4", "lr-1e4"),
            ("wd=1e-5", "wd-1e5"),
            ("wd=0", "wd-0"),
        ]),
        ("L4: 数据增强", [
            ("Base", "sch-cos"),
            ("+Geometric OK", "aug-geo"),
            ("+Color XX", "aug-color"),
        ]),
        ("L5: 损失 + 正则", [
            ("div=5", "aug-geo"),
            ("div=3 OK", "loss-div3"),
            ("div=7", "loss-div7"),
            ("LabelSmooth", "reg-ls01"),
            ("Dropout", "reg-drop03"),
        ]),
        ("L6: 训练配置", [
            ("bs=32/ep=60", "loss-div3"),
            ("bs=64", "cfg-bs64"),
            ("ep=80 OK", "cfg-ep80"),
        ]),
        ("L7: 最终验证", [
            ("seed=42", "cfg-ep80"),
            ("seed=123", "final-123"),
            ("seed=456", "final-456"),
        ]),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()

    for idx, (title, experiments) in enumerate(layers):
        ax = axes[idx]
        labels = [e[0] for e in experiments]
        names = [e[1] for e in experiments]
        accs = []
        stds = []
        for n in names:
            m = load_metrics(n)
            if m:
                accs.append(m['mean_accuracy'] * 100)
                stds.append(m['std_accuracy'] * 100)
            else:
                accs.append(0)
                stds.append(0)

        x = np.arange(len(labels))
        bars = ax.bar(x, accs, 0.55, yerr=stds, capsize=4,
                      color=COLORS[:len(labels)], edgecolor='white', linewidth=0.8,
                      error_kw={'linewidth': 1.2, 'ecolor': '#444444'})

        # 数值标注
        for bar, acc, std in zip(bars, accs, stds):
            if acc > 0:
                color = COLORS[2] if acc == max(accs) else '#333333'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.3,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=7.5,
                        fontweight='bold' if acc == max(accs) else 'normal',
                        color=color)
                if acc < 30:  # 低值特殊标注
                    ax.annotate('XX', (bar.get_x() + bar.get_width()/2, 5),
                                ha='center', fontsize=16)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel('Accuracy (%)', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 105)

        # 高亮最优
        if max(accs) > 0:
            best_idx = accs.index(max(accs))
            bars[best_idx].set_edgecolor(COLORS[2])
            bars[best_idx].set_linewidth(2.5)

    # 隐藏多余 subplot
    axes[-1].set_visible(False)

    fig.suptitle('CAFE 阶段三 — 分层对比总览', fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "layer-comparison.png")
    plt.close(fig)
    print(f"  OK layer-comparison.png")

# ── 3. 最优实验 Loss 曲线 ────────────────────────────
def chart_loss_curves():
    """最优实验 (loss-div3) 的 Train/Val Loss 和 Acc 曲线"""
    data = get_fold_data("loss-div3")
    if not data:
        print("  !! loss-div3 data not found")
        return

    histories = data['fold_histories']

    # 汇总所有 fold 的平均 epoch 数据
    max_epochs = max(len(h['epochs']) for h in histories)
    all_train_loss = np.full((len(histories), max_epochs), np.nan)
    all_val_loss = np.full((len(histories), max_epochs), np.nan)
    all_train_acc = np.full((len(histories), max_epochs), np.nan)
    all_val_acc = np.full((len(histories), max_epochs), np.nan)

    for fi, h in enumerate(histories):
        for ei, ep in enumerate(h['epochs']):
            all_train_loss[fi, ei] = ep['train_loss']
            all_val_loss[fi, ei] = ep['val_loss']
            all_train_acc[fi, ei] = ep['train_acc']
            all_val_acc[fi, ei] = ep['val_acc']

    mean_train_loss = np.nanmean(all_train_loss, axis=0)
    mean_val_loss = np.nanmean(all_val_loss, axis=0)
    std_val_loss = np.nanstd(all_val_loss, axis=0)
    mean_train_acc = np.nanmean(all_train_acc, axis=0)
    mean_val_acc = np.nanmean(all_val_acc, axis=0)
    std_val_acc = np.nanstd(all_val_acc, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = np.arange(1, len(mean_train_loss) + 1)
    # 只显示到所有 fold 都有的 epoch
    min_epochs = min(len(h['epochs']) for h in histories)

    # Loss
    ax1.plot(epochs[:min_epochs], mean_train_loss[:min_epochs], color=COLORS[0],
             linewidth=2, label='Train Loss')
    ax1.plot(epochs[:min_epochs], mean_val_loss[:min_epochs], color=COLORS[2],
             linewidth=2, label='Val Loss')
    ax1.fill_between(epochs[:min_epochs],
                     mean_val_loss[:min_epochs] - std_val_loss[:min_epochs],
                     mean_val_loss[:min_epochs] + std_val_loss[:min_epochs],
                     alpha=0.15, color=COLORS[2])
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.8)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Acc
    ax2.plot(epochs[:min_epochs], mean_train_acc[:min_epochs] * 100, color=COLORS[0],
             linewidth=2, label='Train Acc')
    ax2.plot(epochs[:min_epochs], mean_val_acc[:min_epochs] * 100, color=COLORS[2],
             linewidth=2, label='Val Acc')
    ax2.fill_between(epochs[:min_epochs],
                     (mean_val_acc[:min_epochs] - std_val_acc[:min_epochs]) * 100,
                     (mean_val_acc[:min_epochs] + std_val_acc[:min_epochs]) * 100,
                     alpha=0.15, color=COLORS[2])
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    fig.suptitle(f'最优配置 (RN50+Cosine+Geo+div3) — 10-Fold 平均曲线',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / "best-loss-curves.png")
    plt.close(fig)
    print(f"  OK best-loss-curves.png")

# ── 4. 最终 3-Seed 对比 ─────────────────────────────
def chart_final_seeds():
    """3 个 seed 的 fold 分布箱线图"""
    seeds = [("42", "cfg-ep80"), ("123", "final-123"), ("456", "final-456")]
    all_fold_data = []
    labels = []
    for seed_name, exp_name in seeds:
        m = load_metrics(exp_name)
        if m:
            folds = [v * 100 for v in m['fold_results'].values()]
            all_fold_data.append(folds)
            labels.append(f"seed={seed_name}\n{m['mean_accuracy']*100:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(all_fold_data, labels=labels, patch_artist=True,
                     widths=0.5, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor=COLORS[2], markersize=8))

    for patch, color in zip(bp['boxes'], COLORS[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)

    # 叠加散点
    for i, folds in enumerate(all_fold_data):
        jitter = np.random.normal(0, 0.04, len(folds))
        ax.scatter(np.full(len(folds), i+1) + jitter, folds,
                   color=COLORS[i], alpha=0.6, s=40, zorder=3, edgecolors='white',
                   linewidth=0.5)

    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('最终验证: 3 种子 × 10 Fold 分布', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    # 添加均值线
    overall_mean = np.mean([np.mean(f) for f in all_fold_data])
    ax.axhline(y=overall_mean, color=COLORS[2], linestyle='--', linewidth=2, alpha=0.5)
    ax.annotate(f'30-Fold Mean: {overall_mean:.1f}%', xy=(0.5, overall_mean + 0.5),
                fontsize=10, fontweight='bold', color=COLORS[2], ha='center')

    fig.tight_layout()
    fig.savefig(OUT / "final-3seeds.png")
    plt.close(fig)
    print(f"  OK final-3seeds.png")

# ── 5. 关键实验 Acc+Loss 对比表 ──────────────────────
def chart_summary_table():
    """生成汇总对比图（表格形式嵌入）"""
    experiments = [
        ("baseline", "RN18+MSCeleb", "Adam", "Exp", "2e-4", "HFlip+Erase"),
        ("bb-rn18-imagenet", "RN18+ImageNet", "Adam", "Exp", "2e-4", "HFlip+Erase"),
        ("bb-rn34-msceleb", "RN34+ImageNet", "Adam", "Exp", "2e-4", "HFlip+Erase"),
        ("bb-rn50-msceleb", "RN50+ImageNet", "Adam", "Exp", "2e-4", "HFlip+Erase"),
        ("sch-cos", "RN50+ImageNet", "Adam", "Cosine", "2e-4", "HFlip+Erase"),
        ("aug-geo", "RN50+ImageNet", "Adam", "Cosine", "2e-4", "+Geometric"),
        ("loss-div3", "RN50+ImageNet", "Adam", "Cosine", "2e-4", "+Geometric"),
        ("cfg-ep80", "RN50+ImageNet", "Adam", "Cosine", "2e-4", "+Geometric"),
    ]

    data_rows = []
    best_val_loss = []
    for exp_name, *_ in experiments:
        m = load_metrics(exp_name)
        if m:
            data_rows.append({
                'name': exp_name,
                'acc': m['mean_accuracy'] * 100,
                'std': m['std_accuracy'] * 100,
            })
            # 计算最优 fold 的平均 best val loss
            if 'fold_histories' in m:
                losses = []
                for h in m['fold_histories']:
                    best_epoch = max(h['epochs'], key=lambda e: e['val_acc'])
                    losses.append(best_epoch['val_loss'])
                avg_best_loss = np.mean(losses)
                best_val_loss.append(avg_best_loss)
            else:
                best_val_loss.append(None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Acc 柱状图
    names = [r['name'].replace('-', '\n', 1) for r in data_rows]
    accs = [r['acc'] for r in data_rows]
    stds = [r['std'] for r in data_rows]
    x = np.arange(len(names))

    bars = ax1.bar(x, accs, 0.6, yerr=stds, capsize=4, color=COLORS[:len(names)],
                   edgecolor='white', linewidth=0.8,
                   error_kw={'linewidth': 1.2, 'ecolor': '#444444'})
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('优化路径 Accuracy 对比', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax1.set_ylim(60, 100)

    # Loss 对应柱状图
    ax2.bar(x, best_val_loss, 0.6, color=COLORS[:len(names)],
            edgecolor='white', linewidth=0.8)
    for i, (bar, loss) in enumerate(zip(bars, best_val_loss)):
        if loss:
            ax2.text(bar.get_x() + bar.get_width()/2, loss + 0.01,
                     f'{loss:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7.5)
    ax2.set_ylabel('Best Val Loss', fontsize=11, fontweight='bold')
    ax2.set_title('优化路径 Val Loss 对比', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    # 反转 y 轴让更低的 loss 视觉上也更低
    ax2.invert_yaxis()

    fig.suptitle('CAFE 阶段三 — 优化路径 Acc + Loss 总览', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / "acc-loss-summary.png")
    plt.close(fig)
    print(f"  OK acc-loss-summary.png")


# ── 6. Baseline vs Best 各折对比 ─────────────────────
def chart_baseline_vs_best():
    """baseline vs cfg-ep80 的 fold-by-fold 对比"""
    baseline = load_metrics("baseline")
    best = load_metrics("cfg-ep80")
    if not baseline or not best:
        return

    b_folds = [v * 100 for v in baseline['fold_results'].values()]
    t_folds = [v * 100 for v in best['fold_results'].values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 11)
    w = 0.35

    bars1 = ax.bar(x - w/2, b_folds, w, label=f'Baseline ({baseline["mean_accuracy"]*100:.1f}%)',
                   color=COLORS[1], edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + w/2, t_folds, w, label=f'Optimized ({best["mean_accuracy"]*100:.1f}%)',
                   color=COLORS[0], edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars1, b_folds):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%',
                ha='center', fontsize=7, fontweight='bold', color=COLORS[1])
    for bar, val in zip(bars2, t_folds):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%',
                ha='center', fontsize=7, fontweight='bold', color=COLORS[0])

    ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Baseline vs Optimized — 10-Fold 对比', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.set_ylim(0, 110)

    # 标注提升
    improvements = np.array(t_folds) - np.array(b_folds)
    for i, imp in enumerate(improvements):
        ax.annotate(f'+{imp:.0f}%', (x[i], max(b_folds[i], t_folds[i]) + 5),
                    ha='center', fontsize=7.5, color=COLORS[2], fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT / "baseline-vs-best-folds.png")
    plt.close(fig)
    print(f"  OK baseline-vs-best-folds.png")


# ── Main ────────────────────────────────────────────
if __name__ == "__main__":
    print("生成阶段三图表...")
    chart_optimization_path()
    chart_layer_bars()
    chart_loss_curves()
    chart_final_seeds()
    chart_summary_table()
    chart_baseline_vs_best()
    print(f"\n✅ 全部图表已保存至: {OUT}")

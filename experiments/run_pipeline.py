"""
实验流水线自动化脚本。

用法:
    # 运行全部实验（串行，每个间隔 10 分钟冷却）
    python experiments/run_pipeline.py --all

    # 运行单个实验
    python experiments/run_pipeline.py --exp baseline-kfold

    # 预览（不实际运行）
    python experiments/run_pipeline.py --all --dry-run
"""

import os
import sys
import json
import time
import shutil
import subprocess
import argparse
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAFE_DIR = os.path.join(PROJECT_ROOT, "CAFE", "CAFE")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
TRAINING_OUTPUT = os.path.join(PROJECT_ROOT, "KMU-FED", "output_kmu_fed_clip")
METRICS_FILE = os.path.join(TRAINING_OUTPUT, "training_metrics.json")

# 实验定义
EXPERIMENTS = [
    {
        "id": "baseline-kfold",
        "branch": "exp/baseline-kfold",
        "name": "实验1: KFold+YOLO 基线",
        "description": "原始代码，10-Fold CV（按受试者划分），YOLO 人脸检测",
        "cmd": (
            "cd {cafe_dir} && "
            "python CAFE_KMU-FED_train.py "
            "--cv_method kfold --folds 10 --epochs 60 "
            "--batch_size 32 --lr 0.0002"
        ),
        "clip_layers": None,  # 不修改 CLIP
    },
    {
        "id": "clip-9",
        "branch": "exp/clip-9",
        "name": "实验2: CLIP-9 层",
        "description": "减少 CLIP Transformer 到 9 层",
        "cmd": (
            "cd {cafe_dir} && "
            "python CAFE_KMU-FED_train.py "
            "--cv_method kfold --folds 10 --epochs 60 "
            "--batch_size 32 --lr 0.0002 "
            "--clip_layers 9"
        ),
        "clip_layers": 9,
    },
    {
        "id": "clip-6",
        "branch": "exp/clip-6",
        "name": "实验3: CLIP-6 层",
        "description": "减少 CLIP Transformer 到 6 层",
        "cmd": (
            "cd {cafe_dir} && "
            "python CAFE_KMU-FED_train.py "
            "--cv_method kfold --folds 10 --epochs 60 "
            "--batch_size 32 --lr 0.0002 "
            "--clip_layers 6"
        ),
        "clip_layers": 6,
    },
    {
        "id": "clip-3",
        "branch": "exp/clip-3",
        "name": "实验4: CLIP-3 层",
        "description": "减少 CLIP Transformer 到 3 层",
        "cmd": (
            "cd {cafe_dir} && "
            "python CAFE_KMU-FED_train.py "
            "--cv_method kfold --folds 10 --epochs 60 "
            "--batch_size 32 --lr 0.0002 "
            "--clip_layers 3"
        ),
        "clip_layers": 3,
    },
    {
        "id": "resnet-only",
        "branch": "exp/resnet-only",
        "name": "实验5: 纯 ResNet-18",
        "description": "移除 CLIP 分支，仅用 ResNet-18 特征分类",
        "cmd": (
            "cd {cafe_dir} && "
            "python CAFE_KMU-FED_train.py "
            "--cv_method kfold --folds 10 --epochs 60 "
            "--batch_size 32 --lr 0.0002 "
            "--no_clip"
        ),
        "clip_layers": "none",
    },
    {
        "id": "mobilenet",
        "branch": "exp/mobilenet",
        "name": "实验6: MobileNetV3 替代 CLIP",
        "description": "用 MobileNetV3-Small (~2.5M) 替代 CLIP ViT-B-32 (~151M)",
        "cmd": (
            "cd {cafe_dir} && "
            "python CAFE_KMU-FED_train.py "
            "--cv_method kfold --folds 10 --epochs 60 "
            "--batch_size 32 --lr 0.0002 "
            "--lightweight_encoder mobilenet_v3"
        ),
        "clip_layers": "mobilenet",
    },
    {
        "id": "clip-text",
        "branch": "exp/clip-text",
        "name": "实验7: CLIP 文本编码器辅助",
        "description": "用 CLIP 文本编码器生成表情类别向量，做零样本相似度匹配",
        "cmd": (
            "cd {cafe_dir} && "
            "python CAFE_KMU-FED_train.py "
            "--cv_method kfold --folds 10 --epochs 60 "
            "--batch_size 32 --lr 0.0002 "
            "--clip_text_mode"
        ),
        "clip_layers": "text",
    },
]

COOLING_SECONDS = 600  # 10 分钟


def run_cmd(cmd, dry_run=False):
    """执行命令并返回是否成功。"""
    print(f"\n  🔧 执行: {cmd}")
    if dry_run:
        print("  [DRY RUN] 跳过执行")
        return True
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    return result.returncode == 0


def copy_metrics(exp_id, timestamp):
    """复制训练指标到 experiments/results/ 目录。"""
    if not os.path.exists(METRICS_FILE):
        print(f"  ⚠️ 未找到训练指标文件: {METRICS_FILE}")
        return None

    src = METRICS_FILE
    dst = os.path.join(RESULTS_DIR, f"{exp_id}_{timestamp}.json")
    shutil.copy2(src, dst)
    print(f"  📝 指标已保存: {dst}")

    # 读取并返回
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(exp, dry_run=False):
    """运行单个实验。"""
    print("\n" + "=" * 70)
    print(f"  {exp['name']}")
    print(f"  分支: {exp['branch']}")
    print(f"  说明: {exp['description']}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 创建/切换分支
    branch = exp["branch"]
    if not dry_run:
        # 检查分支是否存在
        result = subprocess.run(
            f"git show-ref --verify --quiet refs/heads/{branch}",
            shell=True, cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            # 分支已存在，切换
            run_cmd(f"git checkout {branch}")
        else:
            # 创建新分支
            run_cmd(f"git checkout -b {branch}")

        # 确保 main 的改动不在工作区
        run_cmd("git status")
    else:
        print(f"  git checkout -b {branch}  # (DRY RUN)")

    # 2. 运行训练
    cmd = exp["cmd"].format(cafe_dir=CAFE_DIR)
    t_start = time.time()
    success = run_cmd(cmd, dry_run=dry_run)
    elapsed = time.time() - t_start

    if not success and not dry_run:
        print(f"  ❌ 训练失败！")
        return None

    print(f"  ⏱️ 训练耗时: {elapsed/60:.1f} 分钟")

    # 3. 收集指标
    metrics = copy_metrics(exp["id"], timestamp)

    # 4. Git 提交并推送
    if not dry_run:
        run_cmd("git add -A")
        run_cmd(
            f'git commit -m "实验完成: {exp["name"]} '
            f'- 准确率: {metrics.get("mean_accuracy", "N/A") if metrics else "N/A"}"'
        )
        run_cmd(f"git push origin {branch}")
        run_cmd("git checkout main")

    return {
        "experiment": exp["id"],
        "name": exp["name"],
        "branch": branch,
        "timestamp": timestamp,
        "metrics": metrics,
        "elapsed_minutes": elapsed / 60,
    }


def update_tracking_file(results):
    """更新 EXPERIMENT_PLAN.md 的进度。"""
    plan_path = os.path.join(PROJECT_ROOT, "EXPERIMENT_PLAN.md")

    if not os.path.exists(plan_path):
        print("  ⚠️ 未找到 EXPERIMENT_PLAN.md")
        return

    with open(plan_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 更新进度表和结果汇总表
    for r in results:
        exp_id = r["experiment"]
        metrics = r.get("metrics") or {}
        acc = metrics.get("mean_accuracy", "N/A")
        std = metrics.get("std_accuracy", "N/A")

        # 替换状态标记（⬜ → ✅）
        # 这需要精确匹配，实际由 learning-guide-updater Agent 处理
        print(f"  📋 {exp_id}: Acc={acc}, Std={std}")

    # 更新时间戳
    content = content.replace(
        "*最后更新：*",
        f"*最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    )

    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="CAFE 轻量化实验流水线")
    parser.add_argument("--all", action="store_true", help="运行全部实验")
    parser.add_argument("--exp", type=str, help="运行指定实验 (ID)")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际运行")
    parser.add_argument("--no-cooling", action="store_true", help="跳过冷却等待")
    parser.add_argument("--cooling", type=int, default=COOLING_SECONDS,
                        help=f"冷却秒数（默认 {COOLING_SECONDS}s = 10分钟）")
    args = parser.parse_args()

    # 确定要运行的实验列表
    if args.exp:
        exp_list = [e for e in EXPERIMENTS if e["id"] == args.exp]
        if not exp_list:
            print(f"❌ 未知实验: {args.exp}")
            print(f"可用实验: {[e['id'] for e in EXPERIMENTS]}")
            return
    elif args.all:
        exp_list = EXPERIMENTS
    else:
        print("请指定 --exp <实验ID> 或 --all")
        print(f"可用实验:")
        for e in EXPERIMENTS:
            print(f"  {e['id']:<20} → {e['name']}")
        return

    # 确保 results 目录存在
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 运行实验
    all_results = []
    for i, exp in enumerate(exp_list):
        print(f"\n{'▼' * 70}")
        print(f"  实验进度: {i+1}/{len(exp_list)}")
        print(f"{'▼' * 70}")

        result = run_experiment(exp, dry_run=args.dry_run)
        if result:
            all_results.append(result)

        # 如果不是最后一个实验，冷却
        if i < len(exp_list) - 1 and not args.dry_run and not args.no_cooling:
            print(f"\n  🌬️ 冷却 {args.cooling/60:.0f} 分钟...")
            print(f"     下一个实验: {exp_list[i+1]['name']}")
            for remaining in range(args.cooling, 0, -30):
                print(f"     {remaining}s...", end="\r")
                time.sleep(30)
            print("     冷却完成！")

    # 更新追踪文件
    if all_results:
        update_tracking_file(all_results)

    # 汇总
    print("\n" + "=" * 70)
    print("  实验结果汇总")
    print("=" * 70)
    for r in all_results:
        m = r.get("metrics") or {}
        acc = m.get("mean_accuracy", "N/A")
        std = m.get("std_accuracy", "N/A")
        print(f"  {r['name']:<30} Acc={acc} ± {std}  ({r['elapsed_minutes']:.0f}min)")

    print(f"\n结果目录: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

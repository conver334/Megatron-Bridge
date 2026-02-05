#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
比较两种权重加载方法的差异

Usage:
    # 比较所有rank的权重
    python compare_weights.py --num-ranks 8

    # 比较特定rank的权重
    python compare_weights.py --weight1 weights_method1_rank0.pt --weight2 weights_method2_rank0.pt
"""

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# 导入分布式张量支持
try:
    import torch.distributed.tensor
except ImportError:
    pass

console = Console()


def load_weights(path: str):
    """加载权重文件"""
    try:
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含DTensor的文件
        weights = torch.load(path, map_location="cpu", weights_only=False)
        console.print(f"[green]成功加载权重文件: {path}[/green]")
        console.print(f"包含 {len(weights)} 个参数")
        return weights
    except Exception as e:
        console.print(f"[red]加载权重文件失败 {path}: {e}[/red]")
        return None


def compare_weights(weights1, weights2, output_file="weight_comparison_report.txt"):
    """对比两个权重字典"""

    console.print("\n[bold cyan]开始对比权重...[/bold cyan]\n")

    # 获取参数名称集合
    keys1 = set(weights1.keys())
    keys2 = set(weights2.keys())

    # 检查参数名称差异
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    common_keys = keys1 & keys2

    console.print(f"方法1特有的参数: {len(only_in_1)}")
    console.print(f"方法2特有的参数: {len(only_in_2)}")
    console.print(f"共同的参数: {len(common_keys)}\n")

    # 创建结果表格
    table = Table(title="权重对比结果", show_header=True, header_style="bold magenta")
    table.add_column("参数名称", style="cyan", no_wrap=False, width=60)
    table.add_column("形状", justify="right", style="green")
    table.add_column("最大绝对差异", justify="right", style="yellow")
    table.add_column("平均绝对差异", justify="right", style="yellow")
    table.add_column("相对误差(%)", justify="right", style="red")
    table.add_column("是否相同", justify="center", style="bold")

    # 统计信息
    stats = {
        "total_params": len(common_keys),
        "identical": 0,
        "different": 0,
        "max_diff": 0,
        "avg_diff": 0,
        "differences": [],
    }

    # 对比共同参数
    for key in sorted(common_keys):
        param1 = weights1[key]
        param2 = weights2[key]

        # 检查形状
        if param1.shape != param2.shape:
            console.print(f"[red]警告: {key} 形状不匹配: {param1.shape} vs {param2.shape}[/red]")
            continue

        # 计算差异，解决 DTensor/设备不一致报错
        # 将参数移动到 CPU 并转为 torch.Tensor，如果是 DTensor
        param1_local = param1.to_local() if hasattr(param1, "to_local") else param1
        param2_local = param2.to_local() if hasattr(param2, "to_local") else param2
        param1_cpu = param1_local.detach().cpu()
        param2_cpu = param2_local.detach().cpu()
        if param1_cpu.numel() == 0 or param2_cpu.numel() == 0:
            console.print(f"[red]警告: {key} 张量大小为0[/red]")
            continue
        diff = torch.abs(param1_cpu - param2_cpu).view(-1)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        # 计算相对误差
        max_val = max(torch.abs(param1).max().item(), torch.abs(param2).max().item())
        relative_error = (max_diff / max_val * 100) if max_val > 1e-10 else 0

        # 判断是否相同（使用较宽松的阈值）
        is_same = torch.allclose(param1_cpu, param2_cpu, rtol=1e-5, atol=1e-8)

        if is_same:
            stats["identical"] += 1
            status = "✓"
        else:
            stats["different"] += 1
            status = "✗"
            stats["differences"].append(
                {"name": key, "max_diff": max_diff, "mean_diff": mean_diff, "relative_error": relative_error}
            )

        stats["max_diff"] = max(stats["max_diff"], max_diff)
        stats["avg_diff"] += mean_diff

        # 只显示前20个参数或有差异的参数
        if len(table.rows) < 20 or not is_same:
            table.add_row(
                key[:60],
                str(list(param1.shape)),
                f"{max_diff:.2e}",
                f"{mean_diff:.2e}",
                f"{relative_error:.4f}",
                status,
            )

    stats["avg_diff"] /= max(stats["total_params"], 1)

    # 显示表格
    console.print(table)

    # 显示统计摘要
    console.print("\n")
    summary = f"""
[bold]对比统计摘要:[/bold]

总参数数量: {stats["total_params"]}
完全相同的参数: {stats["identical"]} ({stats["identical"] / max(stats["total_params"], 1) * 100:.2f}%)
有差异的参数: {stats["different"]} ({stats["different"] / max(stats["total_params"], 1) * 100:.2f}%)
最大绝对差异: {stats["max_diff"]:.2e}
平均绝对差异: {stats["avg_diff"]:.2e}
    """
    console.print(Panel(summary, title="统计结果", border_style="green"))

    # 保存详细报告
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("权重对比详细报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总参数数量: {stats['total_params']}\n")
        f.write(f"完全相同的参数: {stats['identical']}\n")
        f.write(f"有差异的参数: {stats['different']}\n")
        f.write(f"最大绝对差异: {stats['max_diff']:.2e}\n")
        f.write(f"平均绝对差异: {stats['avg_diff']:.2e}\n\n")

        if only_in_1:
            f.write("\n仅在方法1中存在的参数:\n")
            for key in sorted(only_in_1):
                f.write(f"  - {key}\n")

        if only_in_2:
            f.write("\n仅在方法2中存在的参数:\n")
            for key in sorted(only_in_2):
                f.write(f"  - {key}\n")

        if stats["differences"]:
            f.write("\n\n有差异的参数详情:\n")
            f.write("-" * 80 + "\n")
            for diff in sorted(stats["differences"], key=lambda x: x["max_diff"], reverse=True):
                f.write(f"\n参数: {diff['name']}\n")
                f.write(f"  最大差异: {diff['max_diff']:.2e}\n")
                f.write(f"  平均差异: {diff['mean_diff']:.2e}\n")
                f.write(f"  相对误差: {diff['relative_error']:.4f}%\n")

        f.write("\n\n所有共同参数对比:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'参数名称':<60} {'形状':<20} {'最大差异':<15} {'平均差异':<15}\n")
        f.write("-" * 80 + "\n")

        for key in sorted(common_keys):
            param1 = weights1[key]
            param2 = weights2[key]
            if param1.shape == param2.shape:
                diff = torch.abs(param1 - param2)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                f.write(f"{key:<60} {str(list(param1.shape)):<20} {max_diff:<15.2e} {mean_diff:<15.2e}\n")

    console.print(f"\n[green]详细报告已保存到: {output_file}[/green]")

    return stats


def compare_all_ranks(num_ranks: int, output_dir: str = "."):
    """对比所有rank的权重"""
    console.print(f"[bold blue]开始对比所有 {num_ranks} 个rank的权重...[/bold blue]\n")

    all_stats = []
    failed_ranks = []

    for rank in range(num_ranks):
        weight1_file = Path(output_dir) / f"weights_method1_rank{rank}.pt"
        weight2_file = Path(output_dir) / f"weights_method2_rank{rank}.pt"

        console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
        console.print(f"[bold cyan]正在对比 Rank {rank}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 80}[/bold cyan]\n")

        if not weight1_file.exists():
            console.print(f"[red]警告: 文件不存在 {weight1_file}[/red]")
            failed_ranks.append(rank)
            continue

        if not weight2_file.exists():
            console.print(f"[red]警告: 文件不存在 {weight2_file}[/red]")
            failed_ranks.append(rank)
            continue

        weights1 = load_weights(str(weight1_file))
        weights2 = load_weights(str(weight2_file))

        if weights1 is None or weights2 is None:
            console.print(f"[red]错误: 无法加载rank {rank}的权重文件[/red]")
            failed_ranks.append(rank)
            continue

        output_file = Path(output_dir) / f"weight_comparison_rank{rank}_report.txt"
        stats = compare_weights(weights1, weights2, str(output_file))
        stats["rank"] = rank
        all_stats.append(stats)

    # 生成总体报告
    console.print(f"\n[bold blue]{'=' * 80}[/bold blue]")
    console.print("[bold blue]所有Rank对比总结[/bold blue]")
    console.print(f"[bold blue]{'=' * 80}[/bold blue]\n")

    summary_table = Table(title="各Rank对比统计", show_header=True, header_style="bold magenta")
    summary_table.add_column("Rank", justify="center", style="cyan")
    summary_table.add_column("总参数数", justify="right", style="green")
    summary_table.add_column("相同参数", justify="right", style="green")
    summary_table.add_column("不同参数", justify="right", style="red")
    summary_table.add_column("最大差异", justify="right", style="yellow")
    summary_table.add_column("平均差异", justify="right", style="yellow")
    summary_table.add_column("状态", justify="center", style="bold")

    for stats in all_stats:
        if stats["different"] == 0:
            status = "✓ 完全一致"
            status_style = "green"
        elif stats["max_diff"] < 1e-6:
            status = "⚠ 微小差异"
            status_style = "yellow"
        else:
            status = "✗ 明显差异"
            status_style = "red"

        summary_table.add_row(
            str(stats["rank"]),
            str(stats["total_params"]),
            str(stats["identical"]),
            str(stats["different"]),
            f"{stats['max_diff']:.2e}",
            f"{stats['avg_diff']:.2e}",
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(summary_table)

    # 保存总体报告
    summary_file = Path(output_dir) / "weight_comparison_all_ranks_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("所有Rank权重对比总结报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"对比的Rank数量: {len(all_stats)}\n")
        f.write(f"失败的Rank数量: {len(failed_ranks)}\n")
        if failed_ranks:
            f.write(f"失败的Rank: {failed_ranks}\n")
        f.write("\n")

        for stats in all_stats:
            f.write(f"\nRank {stats['rank']}:\n")
            f.write(f"  总参数数: {stats['total_params']}\n")
            f.write(f"  相同参数: {stats['identical']}\n")
            f.write(f"  不同参数: {stats['different']}\n")
            f.write(f"  最大差异: {stats['max_diff']:.2e}\n")
            f.write(f"  平均差异: {stats['avg_diff']:.2e}\n")

    console.print(f"\n[green]总体报告已保存到: {summary_file}[/green]")

    # 总结论
    all_identical = all(s["different"] == 0 for s in all_stats)
    all_minor = all(s["max_diff"] < 1e-6 for s in all_stats)

    if all_identical:
        console.print("\n[bold green]✓ 所有Rank的权重完全一致！[/bold green]")
    elif all_minor:
        console.print("\n[bold yellow]⚠ 所有Rank的权重有微小差异（可能是数值精度问题）[/bold yellow]")
    else:
        console.print("\n[bold red]✗ 某些Rank的权重存在明显差异！[/bold red]")


def main():
    """
    Compare the weights loaded by two methods
    """
    parser = argparse.ArgumentParser(description="比较两种权重加载方法的差异")
    parser.add_argument("--weight1", type=str, default=None, help="方法1保存的权重文件路径（单个文件模式）")
    parser.add_argument("--weight2", type=str, default=None, help="方法2保存的权重文件路径（单个文件模式）")
    parser.add_argument(
        "--output", type=str, default="weight_comparison_report.txt", help="输出报告文件路径（单个文件模式）"
    )
    parser.add_argument("--num-ranks", type=int, default=None, help="对比的rank数量（批量模式）")
    parser.add_argument("--output-dir", type=str, default=".", help="权重文件所在目录（批量模式）")

    args = parser.parse_args()

    console.print("[bold blue]权重对比工具[/bold blue]\n")

    # 批量模式：对比所有rank
    if args.num_ranks is not None:
        compare_all_ranks(args.num_ranks, args.output_dir)
    # 单文件模式：对比指定的两个文件
    elif args.weight1 and args.weight2:
        # 加载权重
        weights1 = load_weights(args.weight1)
        weights2 = load_weights(args.weight2)

        if weights1 is None or weights2 is None:
            console.print("[red]错误: 无法加载权重文件[/red]")
            return

        # 对比权重
        stats = compare_weights(weights1, weights2, args.output)

        # 结论
        if stats["different"] == 0:
            console.print("\n[bold green]✓ 两种方法加载的权重完全一致！[/bold green]")
        elif stats["max_diff"] < 1e-6:
            console.print("\n[bold yellow]⚠ 两种方法加载的权重有微小差异（可能是数值精度问题）[/bold yellow]")
        else:
            console.print("\n[bold red]✗ 两种方法加载的权重存在明显差异！[/bold red]")
    else:
        console.print("[red]错误: 请指定 --num-ranks（批量模式）或 --weight1 和 --weight2（单文件模式）[/red]")
        parser.print_help()


if __name__ == "__main__":
    main()

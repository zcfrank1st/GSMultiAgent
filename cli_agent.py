#!/usr/bin/env python3
"""
Multi-Agent Guidance System - CLI End-to-End Execution
"""

import asyncio
import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from multi_agent.simulation.guidance_simulator import GuidanceParameters

dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

async def main(args):
    print("=" * 80)
    print("Multi-Agent Guidance System - CLI Workflow")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Task: {args.prompt}")
    print("=" * 80)

    # 1. Initialize components
    print("\n[Step 1] Initializing components...")
    
    from multi_agent import (
        HermesIntegration,
        HERMES_AVAILABLE,
        IntelligentTaskPlanner,
        RAGKnowledgeBase,
        DynamicMemoryBuffer,
        MemoryType,
    )
    from multi_agent.simulation import GuidanceSimulator

    rag = RAGKnowledgeBase()
    await rag.initialize()
    # Dummy data just to ensure RAG works
    await rag.index_documents([
        {"content": "比例导引律（PN）是最常用的导引方法，导航系数通常取3-5", "metadata": {"topic": "guidance_law"}},
        {"content": "拦截机动目标的导航系数需要增大到4-6", "metadata": {"topic": "guidance_law"}},
        {"content": "阻尼比0.2-0.5提供良好的系统稳定性", "metadata": {"topic": "stability"}},
    ])
    
    dmb = DynamicMemoryBuffer()
    
    simulator = GuidanceSimulator(
        output_dir="./guidance_output",
        engine="octave"  # Use octave as requested
    )

    hermes = None
    hermes_available = False
    if HERMES_AVAILABLE:
        hermes = HermesIntegration()
        hermes_available = await hermes.initialize()
    
    # 2. Task Planning
    print("\n[Step 2] Task Planning...")
    if hermes and hermes_available:
        planner = IntelligentTaskPlanner(hermes=hermes)
        task_plan = await planner.analyze_and_plan(args.prompt)
        print(f"  Plan: {task_plan.strategy.value} strategy with {task_plan.subagent_count} subtasks")
        print(f"  Reason: {task_plan.reason}")
    else:
        print("  Hermes not available. Skipping task planning.")

    # 3. RAG Knowledge Retrieval
    print("\n[Step 3] RAG Knowledge Retrieval...")
    rag_results = await rag.retrieve("导引律", top_k=2)
    for i, r in enumerate(rag_results, 1):
        print(f"  [{i}] {r['content'][:50]}...")

    # 4. Simulation & Optimization
    print("\n[Step 4] Simulation & Optimization (Octave)...")
    
    # Use baseline parameters
    initial_params = GuidanceParameters(
        navigation_coefficient=3.0,
        damping_ratio=0.3,
        target_position=[20000.0, 2000.0, 5000.0],
        target_velocity=[0.0, 0.0, 0.0],
        initial_position=[0.0, 7000.0, 0.0],
        initial_velocity=[960.0, 0.0, 0.0]
    )

    study_grid = {
        "navigation_coefficient": [3.0, 4.0, 5.0],
        "damping_ratio": [0.2, 0.3, 0.4],
    }

    study_results = await simulator.parameter_study(
        param_grid=study_grid,
        duration=100.0,
        dt=0.01,
    )

    valid_results = [r for r in study_results if r["metrics"]["success"]]
    if not valid_results:
        # If external simulation fails, fallback logic usually catches it, but if it doesn't:
        print("  All simulations failed! Using best effort metrics.")
        valid_results = study_results

    best_result = min(
        valid_results,
        key=lambda r: (r["metrics"]["miss_distance"] * 1.0 + r["metrics"]["control_energy"] * 10.0),
    )

    print(f"  Best Parameters: Nav={best_result['parameters']['navigation_coefficient']}, Damping={best_result['parameters']['damping_ratio']}")
    print(f"  Best Metrics: Miss Distance={best_result['metrics']['miss_distance']:.4f}m, Control Energy={best_result['metrics']['control_energy']:.4f}J")

    # 5. DMB Experience Writeback
    print("\n[Step 5] Writing Experience to DMB...")
    await dmb.store(
        task_context={"task": "guidance_optimization", "prompt": args.prompt},
        parameters=best_result["parameters"],
        objectives={
            "miss_distance": best_result["metrics"]["miss_distance"],
            "control_energy": best_result["metrics"]["control_energy"],
        },
        fitness=1.0 / (1.0 + best_result["metrics"]["miss_distance"]),
        memory_type=MemoryType.LONG_TERM,
    )
    print("  Experience saved successfully.")

    # 6. Generate Report
    print("\n[Step 6] Generating Report...")
    report = generate_report(args.prompt, initial_params, best_result, study_results)
    
    report_path = Path("./guidance_output/report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    print("\nWorkflow Complete!")

def generate_report(user_input: str, params: GuidanceParameters, best: dict, study: list) -> str:
    lines = [
        "# 导引系统优化报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 任务描述",
        f"用户输入: {user_input}",
        "",
        "## 初始参数",
        f"- 导航系数: {params.navigation_coefficient}",
        f"- 阻尼比: {params.damping_ratio}",
        "",
        "## 优化结果",
        f"| 参数 | 值 |",
        f"|------|-----|",
        f"| 导航系数 | {best['parameters']['navigation_coefficient']} |",
        f"| 阻尼比 | {best['parameters']['damping_ratio']} |",
        f"| 脱靶量 | {best['metrics']['miss_distance']:.6f} m |",
        f"| 控制能量 | {best['metrics']['control_energy']:.6f} J |",
        "",
        "## 参数研究结果",
        f"| 导航系数 | 阻尼比 | 脱靶量 | 控制能量 |",
        f"|----------|--------|--------|----------|"
    ]
    for r in study:
        p = r["parameters"]
        m = r["metrics"]
        lines.append(
            f"| {p['navigation_coefficient']} | {p['damping_ratio']} | "
            f"{m['miss_distance']:.4f} | {m['control_energy']:.4f} |"
        )
    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Guidance CLI Agent")
    parser.add_argument("--prompt", type=str, required=True, help="User task description")
    args = parser.parse_args()
    asyncio.run(main(args))

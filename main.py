import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rrt import RRT, BiRRT, RRTStar, ImprovedRRT

START = (1, 1)
GOAL  = (18, 18)

OBSTACLES = [
    (3,  2,  2, 6),
    (7,  0,  2, 5),
    (10, 5,  2, 7),
    (5,  10, 2, 5),
    (13, 10, 3, 4),
    (2,  14, 4, 2),
    (8,  14, 4, 3),
    (15, 2,  2, 5),
]

PLANNERS = {
    "rrt":          lambda: RRT(START, GOAL, OBSTACLES),
    "birrt":        lambda: BiRRT(START, GOAL, OBSTACLES),
    "rrt_star":     lambda: RRTStar(START, GOAL, OBSTACLES),
    "improved_rrt": lambda: ImprovedRRT(START, GOAL, OBSTACLES, step_size=0.3, max_iter=5000),
}


def path_length(path):
    if path is None or len(path) < 2:
        return float('inf')
    return sum(np.hypot(path[i+1][0] - path[i][0],
                        path[i+1][1] - path[i][1])
               for i in range(len(path) - 1))


def run_single(name):
    planner = PLANNERS[name]()
    t0 = time.time()
    path = planner.plan()
    elapsed = time.time() - t0
    if path:
        print(f"[{name.upper()}] Path found | Time: {elapsed:.3f}s | Nodes: {len(planner.tree)} | Length: {path_length(path):.2f}")
        planner.visualize(path, title=name.upper())
    else:
        print(f"[{name.upper()}] No path found in {elapsed:.3f}s")


def run_benchmark(trials=10):
    print(f"\n{'='*60}")
    print(f"  Benchmark - {trials} trials per algorithm")
    print(f"{'='*60}")

    import os
    os.makedirs("results", exist_ok=True)
    results = {}

    for name, factory in PLANNERS.items():
        times, lengths, successes = [], [], 0
        last_planner, last_path = None, None
        for _ in range(trials):
            planner = factory()
            t0 = time.time()
            path = planner.plan()
            elapsed = time.time() - t0
            if path:
                successes += 1
                times.append(elapsed)
                lengths.append(path_length(path))
                last_planner, last_path = planner, path

        rate = successes / trials * 100
        avg_t = np.mean(times) if times else float('nan')
        avg_l = np.mean(lengths) if lengths else float('nan')
        results[name] = {
            "rate": rate,
            "avg_t": avg_t,
            "avg_l": avg_l,
            "planner": last_planner,
            "path": last_path
        }
        print(f"\n  {name.upper()}")
        print(f"    Success Rate : {rate:.0f}%")
        print(f"    Avg Time     : {avg_t:.3f}s")
        print(f"    Avg Length   : {avg_l:.2f}")

    print(f"\n{'='*60}\n")

    # Bar chart comparison
    names = list(results.keys())
    avg_times   = [results[n]["avg_t"] for n in names]
    avg_lengths = [results[n]["avg_l"] for n in names]
    rates       = [results[n]["rate"]  for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("RRT Algorithm Benchmark Comparison", fontsize=15, fontweight='bold')

    axes[0].bar(names, avg_times, color=colors)
    axes[0].set_title("Avg Planning Time (s)")
    axes[0].set_ylabel("Seconds")
    for i, v in enumerate(avg_times):
        axes[0].text(i, v + 0.005, f"{v:.3f}s", ha='center', fontsize=9)

    axes[1].bar(names, avg_lengths, color=colors)
    axes[1].set_title("Avg Path Length")
    axes[1].set_ylabel("Units")
    for i, v in enumerate(avg_lengths):
        axes[1].text(i, v + 0.2, f"{v:.1f}", ha='center', fontsize=9)

    axes[2].bar(names, rates, color=colors)
    axes[2].set_title("Success Rate (%)")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(0, 110)
    for i, v in enumerate(rates):
        axes[2].text(i, v + 1, f"{v:.0f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("results/benchmark_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


    # Side by side path plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14))
    fig2.suptitle("Path Planning Results - All Algorithms", fontsize=15, fontweight='bold')

    for ax, name in zip(axes2.flat, names):
        r = results[name]
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_title(
            name.upper() + " | len=" + str(round(r['avg_l'], 1)) + " t=" + str(round(r['avg_t'], 3)) + "s",
            fontsize=11
        )
        ax.set_aspect('equal')

        for obs in OBSTACLES:
            ox, oy, ow, oh = obs
            ax.add_patch(patches.Rectangle((ox, oy), ow, oh, color='gray', alpha=0.7))

        planner = r["planner"]
        path = r["path"]

        if planner is not None:
            tree = getattr(planner, 'tree', [])
            for node in tree:
                if node.parent:
                    ax.plot(
                        [node.x, node.parent.x],
                        [node.y, node.parent.y],
                        '-', color='lightblue', linewidth=0.4
                    )

        if path is not None:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, '-o', color='red', linewidth=2, markersize=2, label='Path')

        ax.plot(START[0], START[1], 'go', markersize=10, label='Start')
        ax.plot(GOAL[0], GOAL[1], 'r*', markersize=12, label='Goal')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/all_paths.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="RRT Path Planning Benchmarker")
    parser.add_argument("--algo", default="improved_rrt",
                        choices=list(PLANNERS.keys()) + ["all"],
                        help="Algorithm to run")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials for benchmark")
    args = parser.parse_args()

    if args.algo == "all":
        run_benchmark(args.trials)
    else:
        run_single(args.algo)


if __name__ == "__main__":
    main()
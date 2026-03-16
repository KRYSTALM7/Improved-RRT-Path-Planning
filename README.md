# Improved RRT Path Planning

> Implementations and benchmarks of RRT, BiRRT, RRT\*, and an **Improved RRT** using Bridge Test narrow passage detection and KDTree nearest-neighbor search — for 2D robot path planning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Algorithms](https://img.shields.io/badge/Algorithms-4-orange)

---

## Overview

This repository implements and compares four Rapidly-exploring Random Tree (RRT) algorithms for autonomous robot navigation in 2D environments with rectangular obstacles.

| Algorithm | Key Idea |
|-----------|----------|
| **RRT** | Standard random tree expansion |
| **BiRRT** | Bidirectional growth from start & goal simultaneously |
| **RRT\*** | Rewires tree for asymptotically optimal paths |
| **Improved RRT** | Bridge Test narrow passage detection + KDTree + adaptive step size |

The Improved RRT is inspired by research on handling **narrow corridors** — the hardest case for standard RRT — using the Bridge Test to bias sampling toward bottleneck regions.

---

## Project Structure
```
Improved-RRT-Path-Planning/
├── rrt/
│   ├── __init__.py        # Package exports
│   ├── rrt.py             # Basic RRT
│   ├── bi_rrt.py          # Bidirectional RRT
│   ├── rrt_star.py        # RRT* (optimal)
│   └── improved_rrt.py    # Improved RRT (Bridge Test + KDTree)
├── main.py                # Unified entry point + benchmarker
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙Installation
```bash
git clone https://github.com/KRYSTALM7/Improved-RRT-Path-Planning.git
cd Improved-RRT-Path-Planning
pip install -r requirements.txt
```

---

## Usage

**Run a single algorithm (with visualization):**
```bash
python main.py --algo rrt
python main.py --algo birrt
python main.py --algo rrt_star
python main.py --algo improved_rrt
```

**Benchmark all algorithms over N trials:**
```bash
python main.py --algo all --trials 20
```

**Use in your own code:**
```python
from rrt import ImprovedRRT

planner = ImprovedRRT(
    start=(1, 1),
    goal=(18, 18),
    obstacles=[(3, 2, 2, 6), (7, 0, 2, 5)]  # (x, y, width, height)
)
path = planner.plan()
planner.visualize(path)
```

---

## Algorithm Details

### RRT
Incrementally builds a tree by sampling random points and steering the nearest node toward them. Fast but paths are suboptimal and may be jagged.

### BiRRT
Grows two trees — one from the start, one from the goal — and connects them when they meet. Significantly faster than single-tree RRT in open environments.

### RRT\*
Extends RRT with a **rewiring** step: after adding a new node, it checks if nearby nodes would have a shorter path through the new node, and reconnects them if so. Converges to the optimal path given enough iterations.

### Improved RRT
Three key improvements over standard RRT:

1. **Bridge Test Sampling** — Randomly samples two points inside obstacles at a fixed distance apart. Their midpoint, if collision-free, lies in a narrow passage and gets added to the sample set. This dramatically improves exploration of tight corridors.
2. **KDTree Nearest Neighbor** — Uses `scipy.spatial.KDTree` for O(log n) nearest neighbor queries instead of O(n) linear scan, reducing planning time for large trees.
3. **Adaptive Step Size** — Reduces the expansion step when near obstacle centers, preventing the planner from overshooting into walls.

---

## Sample Benchmark Results

*(10 trials, same environment with 8 obstacles)*

| Algorithm | Success Rate | Avg Time | Avg Path Length |
|-----------|-------------|----------|-----------------|
| RRT | ~90% | ~0.4s | ~28.5 |
| BiRRT | ~85% | ~0.3s | ~27.1 |
| RRT\* | ~88% | ~0.9s | ~24.3 |
| Improved RRT | ~95% | ~0.5s | ~25.8 |

> Run `python main.py --algo all --trials 10` to reproduce these on your machine.

---

## Environment Format

Obstacles are defined as `(x, y, width, height)` tuples — axis-aligned rectangles:
```python
OBSTACLES = [
    (3,  2,  2, 6),   # x=3, y=2, width=2, height=6
    (7,  0,  2, 5),
    ...
]
```

---

## Dependencies
```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```

---

## Contributing

PRs welcome! Ideas for extension:
- 3D environment support
- Dynamic obstacle avoidance
- RRT-Connect variant
- Informed RRT*

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Implementations inspired by research on RRT variants and the Bridge Test methodology for narrow passage sampling in robot motion planning.
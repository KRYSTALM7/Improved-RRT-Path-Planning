"""
Improved RRT — Bridge Test + KDTree Nearest Neighbor
Detects narrow passages using Bridge Test sampling and uses
scipy KDTree for O(log n) nearest neighbor search.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import KDTree
from .rrt import Node


class ImprovedRRT:
    def __init__(self, start, goal, obstacles, x_range=(0, 20), y_range=(0, 20),
                 step_size=0.5, max_iter=3000, goal_radius=0.5,
                 bridge_sample_rate=0.2, bridge_length=2.0):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacles = obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_radius = goal_radius
        self.bridge_sample_rate = bridge_sample_rate
        self.bridge_length = bridge_length
        self.tree = [self.start]
        self._coords = [[start[0], start[1]]]
        self._kdtree = KDTree(self._coords)

    def _rebuild_kdtree(self):
        self._coords = [[n.x, n.y] for n in self.tree]
        self._kdtree = KDTree(self._coords)

    def sample(self):
        """Sample with goal bias and Bridge Test for narrow passages."""
        r = np.random.rand()
        if r < 0.1:
            return Node(self.goal.x, self.goal.y)
        elif r < 0.1 + self.bridge_sample_rate:
            return self._bridge_sample()
        else:
            x = np.random.uniform(*self.x_range)
            y = np.random.uniform(*self.y_range)
            return Node(x, y)

    def _bridge_sample(self):
        """
        Bridge Test: sample a point in an obstacle, then sample another
        point in an obstacle at bridge_length away. The midpoint between
        them is likely in a narrow passage — useful for navigation.
        """
        for _ in range(50):
            x1 = np.random.uniform(*self.x_range)
            y1 = np.random.uniform(*self.y_range)
            if not self.in_collision(Node(x1, y1)):
                continue
            angle = np.random.uniform(0, 2 * np.pi)
            x2 = x1 + self.bridge_length * np.cos(angle)
            y2 = y1 + self.bridge_length * np.sin(angle)
            if not self.in_collision(Node(x2, y2)):
                continue
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if not self.in_collision(Node(mx, my)):
                return Node(mx, my)
        # fallback to uniform
        return Node(np.random.uniform(*self.x_range),
                    np.random.uniform(*self.y_range))

    def nearest(self, sampled):
        self._rebuild_kdtree()
        _, idx = self._kdtree.query([sampled.x, sampled.y])
        return self.tree[idx]

    def adaptive_step(self, from_node, to_node):
        """Reduce step size near obstacles for safer expansion."""
        base = self.step_size
        for (ox, oy, ow, oh) in self.obstacles:
            cx, cy = ox + ow / 2, oy + oh / 2
            dist = np.hypot(from_node.x - cx, from_node.y - cy)
            if dist < max(ow, oh):
                base = min(base, self.step_size * 0.5)
        return base

    def steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = np.hypot(dx, dy)
        if dist == 0:
            return None
        step = self.adaptive_step(from_node, to_node)
        scale = min(step, dist) / dist
        new_node = Node(from_node.x + dx * scale, from_node.y + dy * scale)
        new_node.parent = from_node
        return new_node

    def in_collision(self, node):
        for (ox, oy, ow, oh) in self.obstacles:
            if ox <= node.x <= ox + ow and oy <= node.y <= oy + oh:
                return True
        return False

    def edge_in_collision(self, n1, n2, steps=20):
        for i in range(steps + 1):
            t = i / steps
            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)
            if self.in_collision(Node(x, y)):
                return True
        return False

    def reached_goal(self, node):
        return np.hypot(node.x - self.goal.x, node.y - self.goal.y) <= self.goal_radius

    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return list(reversed(path))

    def plan(self):
        for _ in range(self.max_iter):
            sampled = self.sample()
            nearest = self.nearest(sampled)
            new_node = self.steer(nearest, sampled)
            if new_node is None or self.in_collision(new_node):
                continue
            if self.edge_in_collision(nearest, new_node):
                continue
            self.tree.append(new_node)
            if self.reached_goal(new_node):
                goal_node = Node(self.goal.x, self.goal.y)
                goal_node.parent = new_node
                return self.extract_path(goal_node)
        return None

    def visualize(self, path=None, title="Improved RRT (Bridge Test + KDTree)"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(*self.x_range)
        ax.set_ylim(*self.y_range)
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')

        for (ox, oy, ow, oh) in self.obstacles:
            ax.add_patch(patches.Rectangle((ox, oy), ow, oh, color='gray', alpha=0.7))

        for node in self.tree:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y],
                        '-', color='lightblue', linewidth=0.5)

        if path:
            px, py = zip(*path)
            ax.plot(px, py, '-o', color='red', linewidth=2, markersize=3, label='Path')

        ax.plot(self.start.x, self.start.y, 'go', markersize=10, label='Start')
        ax.plot(self.goal.x, self.goal.y, 'r*', markersize=12, label='Goal')
        ax.legend()
        plt.tight_layout()
        plt.show()
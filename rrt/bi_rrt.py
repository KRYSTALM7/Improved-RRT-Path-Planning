"""
BiRRT — Bidirectional Rapidly-exploring Random Tree
Grows two trees simultaneously from start and goal.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .rrt import Node


class BiRRT:
    def __init__(self, start, goal, obstacles, x_range=(0, 20), y_range=(0, 20),
                 step_size=0.5, max_iter=2000, goal_radius=0.5):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacles = obstacles
        self.x_range = x_range
        self.y_range = y_range
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_radius = goal_radius
        self.tree_a = [self.start]
        self.tree_b = [self.goal]

    def sample(self):
        x = np.random.uniform(*self.x_range)
        y = np.random.uniform(*self.y_range)
        return Node(x, y)

    def nearest(self, tree, node):
        dists = [(n, (n.x - node.x)**2 + (n.y - node.y)**2) for n in tree]
        return min(dists, key=lambda d: d[1])[0]

    def steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = np.hypot(dx, dy)
        if dist == 0:
            return None
        scale = min(self.step_size, dist) / dist
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

    def extract_path(self, node_a, node_b):
        path_a = []
        n = node_a
        while n:
            path_a.append((n.x, n.y))
            n = n.parent
        path_a = list(reversed(path_a))

        path_b = []
        n = node_b
        while n:
            path_b.append((n.x, n.y))
            n = n.parent

        return path_a + path_b

    def plan(self):
        for _ in range(self.max_iter):
            sampled = self.sample()

            near_a = self.nearest(self.tree_a, sampled)
            new_a = self.steer(near_a, sampled)
            if new_a is None or self.in_collision(new_a):
                continue
            if self.edge_in_collision(near_a, new_a):
                continue
            self.tree_a.append(new_a)

            near_b = self.nearest(self.tree_b, new_a)
            new_b = self.steer(near_b, new_a)
            if new_b is None or self.in_collision(new_b):
                continue
            if self.edge_in_collision(near_b, new_b):
                continue
            self.tree_b.append(new_b)

            if np.hypot(new_a.x - new_b.x, new_a.y - new_b.y) <= self.step_size:
                return self.extract_path(new_a, new_b)

            self.tree_a, self.tree_b = self.tree_b, self.tree_a

        return None

    def visualize(self, path=None, title="BiRRT"):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(*self.x_range)
        ax.set_ylim(*self.y_range)
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')

        for (ox, oy, ow, oh) in self.obstacles:
            ax.add_patch(patches.Rectangle((ox, oy), ow, oh, color='gray', alpha=0.7))

        for node in self.tree_a:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y],
                        '-', color='lightblue', linewidth=0.5)

        for node in self.tree_b:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y],
                        '-', color='lightyellow', linewidth=0.5)

        if path:
            px, py = zip(*path)
            ax.plot(px, py, '-o', color='red', linewidth=2, markersize=3, label='Path')

        ax.plot(self.start.x, self.start.y, 'go', markersize=10, label='Start')
        ax.plot(self.goal.x, self.goal.y, 'r*', markersize=12, label='Goal')
        ax.legend()
        plt.tight_layout()
        plt.show()
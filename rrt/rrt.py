"""
RRT — Rapidly-exploring Random Tree
Standard implementation for 2D path planning.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class RRT:
    def __init__(self, start, goal, obstacles, x_range=(0, 20), y_range=(0, 20),
                 step_size=0.5, max_iter=2000, goal_radius=0.5):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacles = obstacles  # list of (x, y, width, height)
        self.x_range = x_range
        self.y_range = y_range
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_radius = goal_radius
        self.tree = [self.start]

    def sample(self):
        if np.random.rand() < 0.1:
            return Node(self.goal.x, self.goal.y)
        x = np.random.uniform(*self.x_range)
        y = np.random.uniform(*self.y_range)
        return Node(x, y)

    def nearest(self, sampled):
        dists = [(n, (n.x - sampled.x)**2 + (n.y - sampled.y)**2) for n in self.tree]
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

    def reached_goal(self, node):
        return np.hypot(node.x - self.goal.x, node.y - self.goal.y) <= self.goal_radius

    def extract_path(self, node):
        path = []
        while node is not None:
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

    def visualize(self, path=None, title="RRT"):
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
"""
RRT* — Asymptotically Optimal RRT
Rewires the tree to continuously improve path quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .rrt import Node, RRT


class RRTStar(RRT):
    def __init__(self, *args, rewire_radius=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewire_radius = rewire_radius

    def near_nodes(self, node):
        return [n for n in self.tree
                if np.hypot(n.x - node.x, n.y - node.y) <= self.rewire_radius]

    def cost(self, node):
        c = 0.0
        n = node
        while n.parent:
            c += np.hypot(n.x - n.parent.x, n.y - n.parent.y)
            n = n.parent
        return c

    def plan(self):
        for _ in range(self.max_iter):
            sampled = self.sample()
            nearest = self.nearest(sampled)
            new_node = self.steer(nearest, sampled)
            if new_node is None or self.in_collision(new_node):
                continue
            if self.edge_in_collision(nearest, new_node):
                continue

            # Choose best parent from nearby nodes
            neighbors = self.near_nodes(new_node)
            best_parent = nearest
            best_cost = self.cost(nearest) + np.hypot(
                new_node.x - nearest.x, new_node.y - nearest.y)

            for n in neighbors:
                if self.edge_in_collision(n, new_node):
                    continue
                c = self.cost(n) + np.hypot(new_node.x - n.x, new_node.y - n.y)
                if c < best_cost:
                    best_cost = c
                    best_parent = n

            new_node.parent = best_parent
            self.tree.append(new_node)

            # Rewire nearby nodes through new_node if cheaper
            for n in neighbors:
                if n is best_parent:
                    continue
                if self.edge_in_collision(new_node, n):
                    continue
                new_cost = best_cost + np.hypot(n.x - new_node.x, n.y - new_node.y)
                if new_cost < self.cost(n):
                    n.parent = new_node

            if self.reached_goal(new_node):
                goal_node = Node(self.goal.x, self.goal.y)
                goal_node.parent = new_node
                return self.extract_path(goal_node)

        return None
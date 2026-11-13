"""Simple 5x5 gridworld environment for SARSA algorithm.

Based on the RLCode/reinforcement-learning gridworld.
The agent starts at [0, 0] and tries to reach the goal at [4, 4].
There are two obstacle cells that give -1 reward.
"""

import numpy as np


class Env:
    """5x5 Gridworld with obstacles and a goal."""

    def __init__(self):
        self.width = 5
        self.height = 5
        self.n_actions = 4  # up=0, down=1, left=2, right=3

        # Action effects: [row_delta, col_delta]
        self.actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        # Obstacle and goal positions
        self.obstacles = [[1, 2], [2, 2]]
        self.goal = [4, 4]

        # Rewards
        self.goal_reward = 1.0
        self.obstacle_reward = -1.0
        self.step_reward = 0.0

        self.state = [0, 0]

    def reset(self):
        """Reset the environment to the start state."""
        self.state = [0, 0]
        return list(self.state)

    def step(self, action):
        """Take an action and return (next_state, reward, done)."""
        row, col = self.state

        # Apply action
        new_row = row + self.actions[action][0]
        new_col = col + self.actions[action][1]

        # Boundary check — stay in place if out of bounds
        if 0 <= new_row < self.height and 0 <= new_col < self.width:
            self.state = [new_row, new_col]

        next_state = list(self.state)

        # Check for goal
        if self.state == self.goal:
            return next_state, self.goal_reward, True

        # Check for obstacles
        if self.state in self.obstacles:
            return next_state, self.obstacle_reward, True

        return next_state, self.step_reward, False

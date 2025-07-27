import numpy as np
import torch
from GameNetwork import GameNetwork

class PUCTNode:
    def __init__(self, state, parent=None, prior=1.0):
        self.state = state  # The board state
        self.parent = parent  # Parent node
        self.children = {}  # Child nodes
        self.P = prior  # Prior probability from policy network
        self.N = 0  # Visit count
        self.Q = 0  # Average value

    def is_fully_expanded(self, legal_moves):
        # הצומת נחשב מלא רק אם כל המהלכים החוקיים כבר הורחבו כילדים.
        return len(self.children) == len(legal_moves)

    def select(self, cpuct=1.0):
        """Selects the best action using the PUCT formula."""
        best_score = -float('inf')
        best_action, best_child = None, None
        
        for action, child in self.children.items():
            U = child.Q + cpuct * child.P * np.sqrt(self.N) / (1 + child.N)
            if U > best_score:
                best_score = U
                best_action, best_child = action, child
        
        return best_action, best_child
    
    def expand(self, action, new_state, prior):
        """Expands a new child node."""
        if action not in self.children:
            self.children[action] = PUCTNode(new_state, parent=self, prior=prior)
    
    def update(self, value):
        """Updates the node's statistics."""
        self.N += 1
        self.Q += (value - self.Q) / self.N
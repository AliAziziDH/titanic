import numpy as np
import pandas as pd

class ConfidentSinkhornAllocator:
    def __init__(self, epsilon=0.01, max_iter=100, confidence_threshold=0.90):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold

    def fit_allocate(self, test_probs, survivor_prior=0.3838):
        """
        Allocates labels strictly satisfying marginal distribution constraints
        using entropy-regularized Optimal Transport (Sinkhorn-Knopp).
        """
        N = len(test_probs)
        # Cost matrix: C_i0 = distance to 0 (death), C_i1 = distance to 1 (survival)
        C = np.zeros((N, 2))
        C[:, 0] = test_probs        # Cost of assigning to 0 is high if prob of 1 is high
        C[:, 1] = 1.0 - test_probs  # Cost of assigning to 1 is high if prob of 1 is low

        # Marginals: r is uniform per passenger, c is the historical survivor distribution
        r = np.ones(N) / N
        c = np.array([1.0 - survivor_prior, survivor_prior])

        # Exponentiated Kernel Matrix
        K = np.exp(-C / self.epsilon)
        u = np.ones(N)
        v = np.ones(2)

        for _ in range(self.max_iter):
            u = r / (K @ v)
            v = c / (K.T @ u)

        # Transport Plan
        P = np.diag(u) @ K @ np.diag(v)

        # Soft assignments normalized back to probabilities
        soft_labels = P / P.sum(axis=1, keepdims=True)
        pseudo_probs = soft_labels[:, 1]

        # Extract only high-confidence allocations
        high_conf_idx = np.where((pseudo_probs >= self.confidence_threshold) |
                                 (pseudo_probs <= (1.0 - self.confidence_threshold)))[0]
        pseudo_labels = (pseudo_probs >= 0.5).astype(int)

        return high_conf_idx, pseudo_labels

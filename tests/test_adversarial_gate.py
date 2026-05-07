import unittest
import json
import os
from python.benchmark_gate import evaluate_gate

class TestAdversarialGate(unittest.TestCase):
    def test_block_on_high_overlap(self):
        # Candidate improved from 0.2 to 0.1 (gain 0.1)
        # Adversarial improved from 0.2 to 0.14 (gain 0.06)
        # Overlap = 0.06 / 0.1 = 60% > 50% -> Block
        baseline = {"aggregate_da": 0.2}
        candidate = {"aggregate_da": 0.1}
        adversarial = {"aggregate_da": 0.14}
        
        self.assertFalse(evaluate_gate(candidate, adversarial, baseline))

    def test_pass_on_low_overlap(self):
        # Candidate improved from 0.2 to 0.1 (gain 0.1)
        # Adversarial improved from 0.2 to 0.17 (gain 0.03)
        # Overlap = 0.03 / 0.1 = 30% <= 50% -> Pass
        baseline = {"aggregate_da": 0.2}
        candidate = {"aggregate_da": 0.1}
        adversarial = {"aggregate_da": 0.17}
        
        self.assertTrue(evaluate_gate(candidate, adversarial, baseline))

    def test_block_no_improvement(self):
        baseline = {"aggregate_da": 0.2}
        candidate = {"aggregate_da": 0.25} # Regressed
        adversarial = {"aggregate_da": 0.22}
        
        self.assertFalse(evaluate_gate(candidate, adversarial, baseline))

if __name__ == "__main__":
    unittest.main()

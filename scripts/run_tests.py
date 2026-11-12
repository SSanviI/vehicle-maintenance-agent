"""
Test runner using standard library unittest.
"""
import unittest
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from backend.tests.test_ml import test_predictor_loaded, test_partial_input_prediction, test_preprocessor_from_config
from backend.tests.test_agent import test_agent_state_initialization, test_dont_know_memory_and_no_reasking, test_incremental_prediction_updates


class TestMLAndAgent(unittest.TestCase):

    def test_ml_predictor_loaded(self):
        test_predictor_loaded()

    def test_ml_partial_input(self):
        test_partial_input_prediction()

    def test_ml_preprocessor(self):
        test_preprocessor_from_config()

    def test_agent_state_init(self):
        test_agent_state_initialization()

    def test_dont_know_memory(self):
        test_dont_know_memory_and_no_reasking()

    def test_incremental_prediction(self):
        test_incremental_prediction_updates()


if __name__ == "__main__":
    unittest.main()

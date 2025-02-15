from typing import List, Optional
import pandas as pd
from ..voice_agent import VoiceAgent
from .test_components import TestCase

class VoiceTestRunner:
    def __init__(self, agent: VoiceAgent):
        """
        Initialize the test runner.
        
        Args:
            agent (VoiceAgent): The voice agent to test
        """
        self.agent = agent
        self.test_cases: List[TestCase] = []
        self.current_test: Optional[TestCase] = None
        self.results = []

    def add_test_case(self, test_case: TestCase):
        """Add a test case to the runner"""
        if test_case.validate():
            self.test_cases.append(test_case)
        else:
            raise ValueError("Invalid test case configuration")

    def run_test_case(self, test_case: TestCase) -> dict:
        """Run a single test case"""
        self.current_test = test_case
        
        # Initialize agent connection
        self.agent.initialize_connection()
        
        try:
            # Execute test case and collect results
            results = {
                "test_case": test_case.name,
                "scenario": test_case.scenario.name,
                "metrics_results": {}
            }
            
            # Run metrics evaluation
            for metric in test_case.metrics:
                results["metrics_results"][metric["name"]] = self._evaluate_metric(metric)
            
            return results
            
        finally:
            self.agent.disconnect()
            self.current_test = None

    def run_all_tests(self):
        """Run all test cases"""
        for test_case in self.test_cases:
            result = self.run_test_case(test_case)
            self.results.append(result)

    def generate_test_report(self) -> dict:
        """Generate a comprehensive test report"""
        return {
            "total_tests": len(self.test_cases),
            "completed_tests": len(self.results),
            "results": self.results
        }

    def save_report(self, report_path: str):
        """Save the test report to a CSV file"""
        # Convert results to a pandas DataFrame
        rows = []
        for result in self.results:
            for metric_name, metric_value in result["metrics_results"].items():
                rows.append({
                    "test_case": result["test_case"],
                    "scenario": result["scenario"],
                    "metric": metric_name,
                    "score": metric_value
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(report_path, index=False)

    def _evaluate_metric(self, metric: dict) -> float:
        """Evaluate a single metric"""
        # Implementation for metric evaluation
        pass 
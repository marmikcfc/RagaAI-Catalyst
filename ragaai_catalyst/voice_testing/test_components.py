from typing import List, Dict, Optional

class UserPersona:
    def __init__(self, name: str, prompt: str):
        """
        Initialize a user persona for testing.
        
        Args:
            name (str): Name of the persona
            prompt (str): Description of the persona's characteristics
        """
        self.name = name
        self.prompt = prompt

    def generate_response(self, context: str) -> str:
        """Generate a response based on the persona's characteristics"""
        # Implementation for generating contextual responses
        pass

    def get_persona_attributes(self) -> dict:
        """Get the persona's attributes"""
        return {
            "name": self.name,
            "prompt": self.prompt
        }

class Scenario:
    def __init__(self, name: str, prompt: str):
        """
        Initialize a test scenario.
        
        Args:
            name (str): Name of the scenario
            prompt (str): Description of the scenario including expected flow
        """
        self.name = name
        self.prompt = prompt

class TestCase:
    def __init__(self, name: str, scenario: Scenario, 
                 user_persona: UserPersona, metrics: List[Dict[str, str]]):
        """
        Initialize a test case.
        
        Args:
            name (str): Name of the test case
            scenario (Scenario): Scenario to test
            user_persona (UserPersona): Persona to use for testing
            metrics (List[Dict[str, str]]): Metrics to evaluate
        """
        self.name = name
        self.scenario = scenario
        self.user_persona = user_persona
        self.metrics = metrics
        self.results = None

    def validate(self) -> bool:
        """Validate that the test case is properly configured"""
        return all([
            self.name,
            self.scenario,
            self.user_persona,
            self.metrics
        ])

    def get_test_parameters(self) -> dict:
        """Get all test parameters in a dictionary format"""
        return {
            "name": self.name,
            "scenario": {
                "name": self.scenario.name,
                "prompt": self.scenario.prompt
            },
            "persona": self.user_persona.get_persona_attributes(),
            "metrics": self.metrics
        } 
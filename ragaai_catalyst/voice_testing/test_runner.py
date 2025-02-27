import logging
from typing import List, Optional, Dict
import pandas as pd
from ..voice_agent import Direction, VoiceAgent
from .test_components import TestCase
from .testing_server import TestingServer
from ..voice_agent_evaluation import VoiceAgentEvaluator
import threading
import uvicorn
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceTestRunner:
    def __init__(self, agent: VoiceAgent):
        """
        Initialize the test runner.
        
        Args:
            agent (VoiceAgent): The voice agent to test
            evaluator (VoiceAgentEvaluator): The evaluator to assess conversation quality
        """
        self.agent = agent
        self.test_cases: List[TestCase] = []
        self.current_test: Optional[TestCase] = None
        self.results = []
        self.test_evaluations: Dict[str, dict] = {}
        self.callback_queue = asyncio.Queue()
        self.server_loop = asyncio.new_event_loop()
        self.call_completion_futures: Dict[str, asyncio.Future] = {}
        
        self.server = TestingServer(self.agent, self.callback_queue)
        base_url = self.server.setup_ngrok()
        if not base_url:
            error_msg = "Failed to establish ngrok tunnel during initialization. Cannot proceed."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Create and start the FastAPI application
        app = self.server.create_app()
        
        # Start server in a background thread
        logger.info("Starting server in background thread")
        self.server_thread = threading.Thread(target=self._run_server_with_consumer, args=(app,))
        self.server_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        self.server_thread.start()
        
        # Give server time to start up
        time.sleep(4)
        logger.info(f"Server started at {base_url}")
        # Update agent connection details with server URL
        if hasattr(self.server, 'base_url'):
            self.agent.connection_details['endpoint'] = f"{self.server.base_url}/voice/ws"

    def add_test_case(self, test_case: TestCase):
        """Add a test case to the runner"""
        if test_case.validate():
            self.test_cases.append(test_case)
        else:
            error_msg = "Invalid test case configuration"
            logger.error(error_msg)
            raise ValueError(error_msg)

    async def evaluate_test_case(self, test_case: TestCase, evaluation_data: dict) -> dict:
        """Evaluate a test case using the callback data and transcript"""
        logger.info(f"Evaluating test case: {test_case.name}")
        
        results = {
            "test_case": test_case.name,
            "scenario": test_case.scenario.name,
            "metrics_results": {},
            "transcript": evaluation_data["transcript"],
         
            "evaluator_results": []
        }
        logger.info(f"Evaluating test case: {test_case.name} with evaluator: {test_case.evaluator}")
        test_case.evaluator.evaluate_voice_conversation({"transcript": evaluation_data["transcript"]})
        
        # Run voice agent evaluator
        try:
            results["evaluator_results"]  = await self.evaluator.evaluate_voice_conversation({"transcript": evaluation_data["transcript"]})
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error during voice agent evaluation: {str(e)}")
            results["evaluator_results"] = []
        
        return results

    async def run_test_case(self, test_case: TestCase, time_limit: int = 60) -> dict:
        """Run a single test case"""
        self.current_test = test_case
        
        try:
            # Initialize agent connection for this test case
            self.agent.initialize_connection()
            
            # Set the agent's persona and scenario for this test
            self.agent.set_persona_and_scenario(
                persona=test_case.user_persona.prompt,
                scenario=test_case.scenario.prompt
            )
            
            # Start an outbound call if agent is configured for outbound calls
            if self.agent.direction == Direction.OUTBOUND:
                phone_number = self.agent.get_phone_number()
                if not phone_number:
                    error_msg = "Agent is configured for outbound calls but no phone number is provided"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
                logger.info(f"Initiating outbound call to {phone_number}")
                # Create future before making the call
                call_result = await self.server.start_twilio_call(phone_number, time_limit=time_limit)
                call_sid = call_result.get('call_sid')
                if not call_sid:
                    error_msg = "No call SID returned from Twilio"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Create future for this call
                self.call_completion_futures[call_sid] = asyncio.Future()
                logger.info(f"Call initiated successfully with SID: {call_sid}")
                
                # Wait for call completion and evaluation data
                try:
                    future = self.call_completion_futures[call_sid]
                    evaluation_data = await asyncio.wait_for(future, timeout=60)
                    logger.info(f"Received transcript for call {call_sid}")
                    
                    # Evaluate the test case with the complete data
                    results = await self.evaluate_test_case(test_case, evaluation_data)
                    self.test_evaluations[call_sid] = results
                    # Cleanup
                    del self.call_completion_futures[call_sid]
                    return results
                    
                except asyncio.TimeoutError:
                    error_msg = "Timeout waiting for call completion and evaluation"
                    logger.error(error_msg)
                    if call_sid in self.call_completion_futures:
                        del self.call_completion_futures[call_sid]
                    raise RuntimeError(error_msg)
            
            logger.info(f"Test case '{test_case.name}' completed successfully")
            
        except Exception as e:
            logger.error(f"Error in test case '{test_case.name}': {str(e)}")
            raise
        finally:
            logger.debug("Disconnecting agent")
            self.agent.disconnect()
            self.current_test = None

    def _run_server_with_consumer(self, app, host: str = "0.0.0.0", port: int = 8765):
        """Run the FastAPI server in a separate thread"""
        asyncio.set_event_loop(self.server_loop)
        
        # Schedule the consumer in this event loop
        self.server_loop.create_task(self._post_callback_consumer())
        
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        self.server_loop.run_until_complete(server.serve())

    async def run_all_tests(self, time_limit: int = 20):
        """Run all test cases"""
        logger.info(f"Starting execution of {len(self.test_cases)} test cases")
        
        try:
            # Run tests
            for test_case in self.test_cases:
                logger.info(f"Running test case: {test_case.name}")
                result = await self.run_test_case(test_case, time_limit=time_limit)
                self.results.append(result)
        finally:
            pass
        
        logger.info("All test cases completed")

    async def _evaluate_metric(self, metric: dict, transcript: list) -> float:
        """Evaluate a single metric using the complete transcript"""
        try:
            # Implement metric evaluation logic here
            # This is a placeholder - you should implement your own evaluation logic
            return 0.0
        except Exception as e:
            logger.error(f"Error evaluating metric {metric['name']}: {str(e)}")
            return 0.0

    def generate_test_report(self):
        """Generate a comprehensive test report"""
        logger.info("Generating test report")
        report = {
            "total_tests": len(self.test_cases),
            "completed_tests": len(self.results),
            "results": self.results
        }
        logger.debug(f"Test report generated. Total tests: {report['total_tests']}, Completed: {report['completed_tests']}")
        return report

    def save_report(self, report_path: str):
        """Save the test report to a CSV file"""
        logger.debug(f"Saving test report to: {report_path}")
        try:
            # Convert results to a pandas DataFrame
            rows = []
            logger.info(f"Saving report for {self.results} results")
            for result in self.results:
                # Add metric results
                for metric_name, metric_value in result["metrics_results"].items():
                    rows.append({
                        "test_case": result["test_case"],
                        "scenario": result["scenario"],
                        "type": "metric",
                        "name": metric_name,
                        "result": metric_value,
                        "reason": None
                    })
                
                # Add evaluator results
                for eval_result in result.get("evaluator_results", []):
                    rows.append({
                        "test_case": result["test_case"],
                        "scenario": result["scenario"],
                        "type": "evaluator",
                        "name": eval_result["name"],
                        "result": eval_result["result"],
                        "reason": eval_result["reason"]
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(report_path, index=False)
        except Exception as e:
            logger.error(f"Error saving test report: {str(e)}")
            raise

    async def _post_callback_consumer(self):
        """Continuously process callback data from the callback queue and set the results for associated call SID."""
        while True:
            logger.info("Waiting for callback data from the callback queue")
            callback_data = await self.callback_queue.get()
            logger.info(f"Received enqueued data: {callback_data}")
            call_sid = callback_data.get('CallSid')  # Twilio uses CallSid
            if call_sid in self.call_completion_futures:
                future = self.call_completion_futures[call_sid]
                if not future.done():
                    # Get the transcript for this call from server
                    transcript = self.agent.get_transcript()
                    evaluation_data = {
                        "transcript": transcript
                    }
                    future.set_result(evaluation_data)
                    logger.info(f"Callback processed for call SID: {call_sid}")
                else:
                    logger.warning(f"Future already completed for call SID: {call_sid}")
            else:
                logger.warning(f"Received callback for unknown call SID: {call_sid}") 
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
from fastapi.responses import PlainTextResponse
import json
from twilio.rest import Client
import logging
from enum import Enum
import os
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.processors.transcript_processor import TranscriptProcessor
from openai.types.chat import ChatCompletionToolParam
from dataclasses import dataclass
from typing import Literal, Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Direction(Enum):
    """Enum for call direction"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"

@dataclass
class TranscriptionMessage:
    role: Literal["user", "assistant"]
    content: str
    timestamp: str | None = None

class TranscriptHandler:
    def __init__(self):
        self.messages = []

    async def on_transcript_update(self, processor, frame):
        self.messages.extend(frame.messages)
        for msg in frame.messages:
            timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
            logger.info(f"{timestamp}{msg.role}: {msg.content}")
    
    def get_messages(self):
        return self.messages

class VoiceAgent:
    def __init__(self, agent_id: str, agent_type: str, connection_details: dict, direction: Direction = None):
        """
        Initialize a voice agent for testing.
        
        Args:
            agent_id (str): Unique identifier for the agent
            agent_type (str): Type of agent ("webrtc" or "phone")
            connection_details (dict): Connection configuration for the agent
                For phone type agents, should include:
                - phone_number: The phone number to use for outbound calls
                - endpoint: The WebSocket endpoint for the agent
            direction (Direction): Call direction (INBOUND or OUTBOUND)
        """
        logger.info(f"Initializing VoiceAgent with ID: {agent_id}")
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.connection_details = connection_details
        self.direction = direction
        self.connection = None
        self.persona = None
        self.scenario = None
        self.transcript_handler = TranscriptHandler()
        
        # Validate connection details for phone type agents
        if self.agent_type == "phone" and direction == Direction.OUTBOUND:
            if not connection_details.get('phone_number'):
                error_msg = "Phone number must be provided in connection_details for outbound phone agents"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Initialize services
        self.llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        self.stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True)
        self.tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
            push_silence_after_stop=True,
        )
        
        logger.info(f"VoiceAgent initialized with type: {agent_type}, direction: {direction.value if direction else None}")

    def set_persona_and_scenario(self, persona: str, scenario: str):
        """Set the agent's persona and scenario"""
        self.persona = persona
        self.scenario = scenario

    async def handle_websocket_connection(self, websocket: WebSocket, stream_sid: str, call_sid: str = None):
        """Handle WebSocket connection for the agent"""
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
                serializer=TwilioFrameSerializer(stream_sid),
            ),
        )

        transcript = TranscriptProcessor()
        
        # Set up the agent's context with persona and scenario
        system_prompt = f"""You are an AI agent with the following persona:
        {self.persona}

        Current scenario:
        {self.scenario}

        Your responses will be converted to audio, so avoid using special characters."""

        messages = [{"role": "system", "content": system_prompt}]
        context = OpenAILLMContext(messages)
        context_aggregator = self.llm.create_context_aggregator(context)

        audiobuffer = AudioBufferProcessor(user_continuous_stream=False)

        pipeline = Pipeline([
            transport.input(),
            self.stt,
            transcript.user(),
            context_aggregator.user(),
            self.llm,
            self.tts,
            transport.output(),
            transcript.assistant(),
            audiobuffer,
            context_aggregator.assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
                allow_interruptions=True,
                enable_metrics=True,
                enable_metrics_logging=True,
                enable_usage_metrics=True
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await audiobuffer.start_recording()
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.cancel()

        @transcript.event_handler("on_transcript_update")
        async def handle_update(processor, frame):
            await self.transcript_handler.on_transcript_update(processor, frame)

        runner = PipelineRunner(handle_sigint=False, force_gc=True)
        await runner.run(task)

    def initialize_connection(self):
        """Establish connection with the agent"""
        logger.info(f"Initializing connection for agent {self.agent_id}")
        try:
            if self.agent_type == "webrtc":
                logger.info("Setting up WebRTC connection")
                # WebRTC connection setup will be handled by handle_websocket_connection
                pass
            elif self.agent_type == "phone":
                logger.info("Setting up phone connection")
                # Phone connection setup
                pass
            else:
                error_msg = f"Unsupported agent type: {self.agent_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection: {str(e)}")
            raise
            
    def disconnect(self):
        """Close the connection with the agent"""
        logger.info(f"Disconnecting agent {self.agent_id}")
        if self.connection:
            try:
                self.connection = None
                logger.info("Agent disconnected successfully")
            except Exception as e:
                logger.error(f"Error during disconnection: {str(e)}")
                raise

    def get_status(self):
        """Get the current status of the agent connection"""
        logger.info(f"Getting status for agent {self.agent_id}")
        status = {
            "connected": bool(self.connection),
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "direction": self.direction.value if self.direction else None,
            "persona": self.persona,
            "scenario": self.scenario
        }
        logger.info(f"Agent status: {status}")
        return status

    def get_transcript(self):
        """Get the conversation transcript"""
        return self.transcript_handler.get_messages()

    def make_call(self, phone_number: str):
        """
        Make an outbound call
        
        Args:
            phone_number (str): The phone number to call
        """
        if self.direction != Direction.OUTBOUND:
            error_msg = "Cannot make outbound call with non-outbound agent"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Initiating outbound call to {phone_number}")
        try:
            # Implement call logic here
            logger.info("Call initiated successfully")
            return {"status": "initiated", "phone_number": phone_number}
        except Exception as e:
            logger.error(f"Failed to make call: {str(e)}")
            raise

    def get_call_status(self, call_sid: str):
        """
        Get status of a specific call
        
        Args:
            call_sid (str): The call SID to check
        """
        logger.info(f"Checking status for call {call_sid}")
        try:
            # Implement status check logic here
            status = {"status": "unknown"}  # Placeholder
            logger.debug(f"Call status retrieved: {status}")
            return status
        except Exception as e:
            logger.error(f"Failed to get call status: {str(e)}")
            raise

    def end_call(self, call_sid: str):
        """
        End a specific call
        
        Args:
            call_sid (str): The call SID to end
        """
        logger.info(f"Ending call {call_sid}")
        try:
            # Implement call ending logic here
            logger.debug("Call ended successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to end call: {str(e)}")
            raise

    def get_phone_number(self) -> Optional[str]:
        """Get the phone number configured for this agent"""
        return self.connection_details.get('phone_number')
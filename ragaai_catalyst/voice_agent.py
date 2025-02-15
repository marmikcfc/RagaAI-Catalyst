class VoiceAgent:
    def __init__(self, agent_id: str, agent_type: str, connection_details: dict):
        """
        Initialize a voice agent for testing.
        
        Args:
            agent_id (str): Unique identifier for the agent
            agent_type (str): Type of agent ("webrtc" or "phone")
            connection_details (dict): Connection configuration for the agent
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.connection_details = connection_details
        self.connection = None

    def initialize_connection(self):
        """Establish connection with the agent"""
        if self.agent_type == "webrtc":
            # Initialize WebRTC connection
            pass
        elif self.agent_type == "phone":
            # Initialize phone connection
            pass
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")

    def disconnect(self):
        """Close the connection with the agent"""
        if self.connection:
            # Close connection
            self.connection = None

    def get_status(self):
        """Get the current status of the agent connection"""
        return {
            "connected": bool(self.connection),
            "agent_type": self.agent_type,
            "agent_id": self.agent_id
        } 
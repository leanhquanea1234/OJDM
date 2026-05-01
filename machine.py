from abc import ABC, abstractmethod

class IPiNode(ABC):
    """Interface for handling camera capture and output peripherals."""
    @abstractmethod
    def stream_video(self):
        """Thread 1: Captures and sends video frames."""
        pass

    @abstractmethod
    def process_feedback(self):
        """Thread 2: Receives data back to trigger speaker/monitor."""
        pass



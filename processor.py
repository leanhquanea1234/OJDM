from abc import ABC, abstractmethod
import numpy as np

class IDetector(ABC):
    """Interface for detecting objects using YOLO."""
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """Returns a list of detections (bounding boxes, confidence, etc.)."""
        pass

class IDecider(ABC):
    """Interface for determining actions based on detection results."""
    @abstractmethod
    def evaluate(self, detections: list) -> dict:
        """Decides if an action is needed (e.g., sound or video feedback)."""
        pass

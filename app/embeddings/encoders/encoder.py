from abc import ABC, abstractmethod


class Encoder(ABC):
    """Abstract class for encoders."""

    @abstractmethod
    def text_encode(self, text):
        pass

    @abstractmethod
    def image_encode(self, path):
        pass

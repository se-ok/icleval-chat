import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel


class Shot(BaseModel):
    user: str
    assistant: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ICLSample(ABC, BaseModel):
    uid: int
    task: str
    task_type: str
    label: str
    max_tokens: int
    stop: str

    @abstractmethod
    def format_completion(
        self, input_marker: str | None, output_marker: str | None, newline_after_marker: bool
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_original(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_chat(
        self, input_marker: str | None, output_marker: str | None, newline_after_marker: bool
    ) -> list[Message]:
        raise NotImplementedError

    @abstractmethod
    def check(self, answer: str) -> bool:
        raise NotImplementedError


class ICLEvalTask(ABC):
    filename: Path
    samples: list[ICLSample]

    @abstractmethod
    def load_sample(self, sample: dict) -> ICLSample:
        raise NotImplementedError

    def load(self, path: Path):
        with open(path) as f:
            raw = json.load(f)

        self.samples = [self.load_sample(sample) for sample in raw]
        msg = f"{self.__class__.__qualname__} loaded {len(self.samples)} samples from {self.path}"
        logger.info(msg)

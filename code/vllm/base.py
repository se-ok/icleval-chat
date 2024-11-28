import asyncio
from abc import ABC, abstractmethod
from typing import Literal

import aiohttp
from pydantic import BaseModel

MARKER_QA = ("Question:", "Answer:")
MARKER_QR = ("Question:", "Response:")
MARKER_IO = ("Input:", "Output:")

SHOT_SEPARATOR = "\n\n"
INOUT_SEPARATOR = "\n"


class Shot(BaseModel):
    user: str
    assistant: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


def apply_input_marker(user: str, marker: str | None, newline_after_marker: bool) -> str:
    if marker is None:
        return user

    separator = "\n" if newline_after_marker else " "

    return separator.join((marker, user))


def apply_output_marker(assistant: str, marker: str | None) -> str:
    if marker is None:
        return assistant

    return " ".join((marker, assistant))


class GenericSample(BaseModel):
    uid: int
    task: str
    task_type: str
    shots: list[Shot]
    prompt_shot: str
    prompt_type: Literal["qa", "qr", "io"]
    label: str
    newline_after_marker: bool

    def format_pretrained(
        self, input_marker: str | None, output_marker: str | None, newline_after_marker: bool
    ) -> str:
        shots = []
        for shot in self.shots:
            input_ = apply_input_marker(shot.user, input_marker, newline_after_marker)
            output = apply_output_marker(shot.assistant, output_marker)

            shots.append(INOUT_SEPARATOR.join((input_, output)))

        prompt = apply_input_marker(self.prompt_shot, input_marker, newline_after_marker)
        prompt_out = output_marker if output_marker else ""
        shots.append(prompt + INOUT_SEPARATOR + prompt_out)

        return SHOT_SEPARATOR.join(shots)

    def format_original(self) -> str:
        if self.prompt_type == "qa":
            markers = MARKER_QA
        elif self.prompt_type == "qr":
            markers = MARKER_QR
        elif self.prompt_type == "io":
            markers = MARKER_IO

        input_marker, output_marker = markers

        return self.format_pretrained(
            input_marker=input_marker,
            output_marker=output_marker,
            newline_after_marker=self.newline_after_marker,
        )

    def format_chat(
        self, input_marker: str | None, output_marker: str | None, newline_after_marker: bool
    ) -> list[Message]:
        fewshot = []
        for shot in self.shots:
            user = apply_input_marker(shot.user, input_marker, newline_after_marker)
            fewshot.append(Message(role="user", content=user))

            assistant = apply_output_marker(shot.assistant, output_marker)
            fewshot.append(Message(role="assistant", content=assistant))

        prompt = apply_input_marker(self.prompt_shot, input_marker)
        fewshot.append(Message(role="user", content=prompt))

        return fewshot

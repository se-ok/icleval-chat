import json
import os
import re
import shutil
from pathlib import Path
from typing import Literal

from loguru import logger

MARKER_QA = ("Question:", "Answer:")
MARKER_QR = ("Question:", "Response:")
MARKER_IO = ("Input:", "Output:")


def get_marker(prompt: str) -> Literal["qa", "qr", "io"]:
    prompt_stripped = prompt.strip()

    if prompt.startswith("Input:"):
        assert prompt_stripped.endswith("Output:")
        return "io"

    assert prompt.startswith("Question:")
    if prompt_stripped.endswith("Answer:"):
        return "qa"

    assert prompt_stripped.endswith("Response:")
    return "qr"


def is_newline_after_marker(prompt: str) -> bool:
    for marker in ("Input:", "Question:"):
        if prompt.startswith(marker):
            return prompt[len(marker)] == "\n"

    raise ValueError(f"Valid marker not found in prompt:\n{prompt}")


def separate_examples(examples: str, marker: Literal["qa", "qr", "io"]) -> list[dict]:
    """Return [{'user': ..., 'assistant': ...}, ...], newline after marker"""

    if marker == "qa":
        former, latter = MARKER_QA
    elif marker == "qr":
        former, latter = MARKER_QR
    elif marker == "io":
        former, latter = MARKER_IO

    assert examples.startswith(former)

    separated = []

    for chunk in examples[len(former) :].split(former):
        assert len(re.findall(latter, chunk)) == 1
        user, assistant = chunk.split(latter)

        separated.append({"user": user.strip(), "assistant": assistant.strip()})

    return separated


def separate_prompt(prompt: str, marker: Literal["qa", "qr", "io"]) -> str:
    if marker == "qa":
        former, latter = MARKER_QA
    elif marker == "qr":
        former, latter = MARKER_QR
    elif marker == "io":
        former, latter = MARKER_IO

    assert prompt.startswith(former)
    prompt = prompt[len(former) :].strip()
    assert prompt.find(former) < 0
    assert len(re.findall(latter, prompt)) == 1
    assert prompt.endswith(latter)

    return prompt[: -len(latter)].strip()


def process_sample(sample: dict) -> dict:
    copied = sample.copy()
    if "exmaples" in copied:
        copied["examples"] = copied.pop("exmaples")

    prompt_type = get_marker(copied["prompt"])
    newline_after_marker: bool = is_newline_after_marker(copied["prompt"])

    copied["shots"] = separate_examples(copied["examples"], marker=prompt_type)
    copied["prompt_shot"] = separate_prompt(copied["prompt"], marker=prompt_type)
    copied["prompt_type"] = prompt_type
    copied["newline_after_marker"] = newline_after_marker

    if copied["label"] is True:
        copied["label"] = str(True)

    return copied


for file in os.listdir("."):
    logger.info(f"At file {file}")
    if not file.endswith(".json"):
        logger.info("Not JSON. continue")
        continue

    target = Path("processed") / file

    if file.startswith("copy_"):
        shutil.copy(file, target)
        logger.info(f"Copied {file} to {target}")
        continue

    with open(file) as f:
        original = json.load(f)

    processed = [process_sample(sample) for sample in original]
    with open(target, "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    logger.info(f"Processed {file} to {target}")

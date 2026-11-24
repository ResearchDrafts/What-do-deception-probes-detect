"""
ContextProbeDataset: the 2x2 context x response deception dataset, in Apollo's DialogueDataset format.

Put this file at deception_detection/data/context_probe.py and the CSV at
data/context_probe/Final_Probe_Dataset.csv (relative to the repo root).

Each CSV row becomes one dialogue:
    system:    system_prompt
    user/...:  each "[user] ..." / "[assistant] ..." line of `context` (detect=False)
               (the question is appended to the last user message if it isn't already in it,
                which is the case for S0001-S0020)
    assistant: response (detect=True)  <- the only message the probe scores
"""

import csv
import re
from typing import Any

from deception_detection.data.base import DialogueDataset, DialogueDataType
from deception_detection.types import Dialogue, Label, Message
from deception_detection.utils import REPO_ROOT

CSV_PATH = REPO_ROOT / "data/context_probe/Final_Probe_Dataset.csv"
ROLE_LINE = re.compile(r"^\[(user|assistant)\] (.*)$")
METADATA_COLS = ["row_id", "scenario_id", "cell", "variant", "context_type",
                 "response_label", "response", "domain"]


def row_to_messages(row: dict[str, str]) -> list[tuple[str, str, bool]]:
    """Pure-python conversion of one CSV row to (role, content, detect) triples."""
    msgs: list[tuple[str, str, bool]] = [("system", row["system_prompt"], False)]
    for line in row["context"].split("\n"):
        if not line.strip():
            continue
        m = ROLE_LINE.match(line)
        if m is None:
            raise ValueError(f"{row['row_id']}: context line without [user]/[assistant] tag: {line!r}")
        msgs.append((m.group(1), m.group(2), False))

    if msgs[-1][0] != "user":
        raise ValueError(f"{row['row_id']}: context must end with a user message")
    if row["question"] not in row["context"]:
        role, content, _ = msgs[-1]
        msgs[-1] = (role, f"{content} {row['question']}", False)

    msgs.append(("assistant", row["response"], True))
    return msgs


def load_rows(path=CSV_PATH) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ContextProbeDataset(DialogueDataset):
    base_name = "context_probe"
    all_variants: list[str] = ["plain"]
    default_variant = "plain"
    # padding left at the class default: llama {"left": 4, "right": 1}, same as Apollo's eval sets

    def _get_dialogues(self) -> DialogueDataType:
        dialogues: list[Dialogue] = []
        labels: list[Label] = []
        metadata: dict[str, list[Any]] = {c: [] for c in METADATA_COLS}
        for row in load_rows():
            dialogues.append([Message(role=r, content=c, detect=d)
                              for r, c, d in row_to_messages(row)])
            labels.append(Label.DECEPTIVE if row["response_label"] == "deceptive" else Label.HONEST)
            for c in METADATA_COLS:
                metadata[c].append(row[c])
        return dialogues, labels, metadata

    @staticmethod
    def get_grading_prompt(input_dialogue, response, metadata=None) -> dict[str, str]:
        raise NotImplementedError("Responses are prewritten; no grading needed")

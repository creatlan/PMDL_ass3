"""
Text preprocessing utilities for text-to-image pipelines.

This module focuses on the first stage of the assignment: taking raw prompts
and converting them into normalized strings and tokenized representations ready
for conditioning a diffusion model. The design favors explicit, readable steps
so you can validate each transformation before training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, asdict
from typing import Iterable, List, Sequence

import transformers


@dataclass
class PromptRecord:
    """Container for a single prompt and its processed representations."""

    original: str
    cleaned: str
    tokens: List[str]
    token_ids: List[int]
    attention_mask: List[int]


class PromptPreprocessor:
    """Normalize and tokenize text prompts.

    Parameters
    ----------
    tokenizer_name:
        Hugging Face tokenizer identifier that matches the downstream text
        encoder (e.g., ``"openai/clip-vit-base-patch32"``).
    max_length:
        Maximum sequence length for tokenization. Tokens beyond this length are
        truncated to avoid exceeding the model's context window.
    lowercase:
        Whether to lowercase prompts during cleaning. Useful for models trained
        on lowercased corpora.
    strip_accents:
        Whether to strip accents when the tokenizer supports it.
    """

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int = 77,
        lowercase: bool = True,
        strip_accents: bool | None = None,
    ) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True
        )
        if strip_accents is not None and hasattr(self.tokenizer, "backend_tokenizer"):
            backend = self.tokenizer.backend_tokenizer
            normalizer = backend.normalizer
            if hasattr(normalizer, "strip_accents"):
                normalizer.strip_accents = strip_accents
        self.max_length = max_length
        self.lowercase = lowercase

    def clean_prompt(self, prompt: str) -> str:
        """Apply lightweight, inspection-friendly cleaning to a prompt."""

        cleaned = prompt.strip()
        if self.lowercase:
            cleaned = cleaned.lower()
        # Collapse repeated whitespace to a single space while preserving tokens.
        cleaned = " ".join(cleaned.split())
        return cleaned

    def preprocess_prompt(self, prompt: str) -> PromptRecord:
        """Clean and tokenize a single prompt, returning rich metadata."""

        cleaned = self.clean_prompt(prompt)
        encoded = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors=None,
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        return PromptRecord(
            original=prompt,
            cleaned=cleaned,
            tokens=tokens,
            token_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    def preprocess_prompts(self, prompts: Sequence[str]) -> List[PromptRecord]:
        """Vectorized convenience wrapper for multiple prompts."""

        return [self.preprocess_prompt(prompt) for prompt in prompts]


def load_prompts(path: pathlib.Path) -> List[str]:
    """Load prompts from a text (one per line) or JSONL file.

    JSONL rows must contain a ``"text"`` field.
    """

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        prompts: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "text" not in record:
                    raise KeyError("JSONL rows must contain a 'text' field")
                prompts.append(str(record["text"]))
        return prompts

    # Fallback: treat the file as plain text, one prompt per line.
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def dump_prompt_records(records: Iterable[PromptRecord], output_path: pathlib.Path) -> None:
    """Write preprocessed prompts to a JSONL file for inspection."""

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            json_record = asdict(record)
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and tokenize prompts for text-to-image diffusion training"
    )
    parser.add_argument(
        "--tokenizer",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face tokenizer identifier",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=77,
        help="Maximum token sequence length (padding/truncation applied)",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=False,
        help="Optional path to a .txt or .jsonl file containing prompts",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=[],
        help="Inline prompts to preprocess if no input file is given",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("preprocessed_prompts.jsonl"),
        help="Where to write the JSONL output",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing during cleaning",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts: List[str] = []

    if args.input is not None:
        prompts.extend(load_prompts(args.input))
    if args.prompts:
        prompts.extend(args.prompts)

    if not prompts:
        raise ValueError("No prompts provided. Use --input or --prompts to supply text.")

    preprocessor = PromptPreprocessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        lowercase=not args.no_lowercase,
    )
    records = preprocessor.preprocess_prompts(prompts)
    dump_prompt_records(records, args.output)

    print(f"Processed {len(records)} prompts → {args.output}")


if __name__ == "__main__":
    main()

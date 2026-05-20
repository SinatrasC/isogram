from __future__ import annotations

from typing import Any

import pandas as pd


PERMISSIVE_SOURCE_METADATA: dict[str, dict[str, str]] = {
    "chat_gpt_moth": {
        "source_dataset": "alejopaullier/daigt-external-dataset",
        "source_license": "mit",
        "upstream_url": "https://www.kaggle.com/datasets/alejopaullier/daigt-external-dataset",
    },
    "falcon_180b_v1": {
        "source_dataset": "nbroad/daigt-data-llama-70b-and-falcon180b",
        "source_license": "apache-2.0",
        "upstream_url": "https://www.kaggle.com/datasets/nbroad/daigt-data-llama-70b-and-falcon180b",
    },
    "kingki19_palm": {
        "source_dataset": "kingki19/llm-generated-essay-using-palm-from-google-gen-ai",
        "source_license": "cc0-1.0",
        "upstream_url": "https://www.kaggle.com/datasets/kingki19/llm-generated-essay-using-palm-from-google-gen-ai",
    },
    "palm-text-bison1": {
        "source_dataset": "kingki19/llm-generated-essay-using-palm-from-google-gen-ai",
        "source_license": "cc0-1.0",
        "upstream_url": "https://www.kaggle.com/datasets/kingki19/llm-generated-essay-using-palm-from-google-gen-ai",
    },
    "radek_500": {
        "source_dataset": "radek1/llm-generated-essays",
        "source_license": "cc0-1.0",
        "upstream_url": "https://www.kaggle.com/datasets/radek1/llm-generated-essays",
    },
    "radekgpt4": {
        "source_dataset": "radek1/llm-generated-essays",
        "source_license": "cc0-1.0",
        "upstream_url": "https://www.kaggle.com/datasets/radek1/llm-generated-essays",
    },
}

EXCLUDED_SOURCE_REASONS: dict[str, str] = {
    "persuade_corpus": "cc-by-nc-sa-4.0",
    "mistral7binstruct_v2": "unverified",
    "llama2_chat": "llama-license",
    "mistral7binstruct_v1": "unverified",
    "train_essays": "competition-release",
    "llama_70b_v1": "llama-license",
    "darragh_claude_v6": "unknown-license",
    "darragh_claude_v7": "unknown-license",
    "NousResearch/Llama-2-7b-chat-hf": "llama-license",
    "mistralai/Mistral-7B-Instruct-v0.1": "unverified",
    "cohere-command": "unverified",
}

LICENSE_MAPPED_SOURCES = frozenset(PERMISSIVE_SOURCE_METADATA) | frozenset(EXCLUDED_SOURCE_REASONS)


def filter_permissive_source_rows(
    frame: pd.DataFrame,
    *,
    source_column: str = "source",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source_column not in frame.columns:
        return frame.copy(), {
            "enabled": False,
            "reason": f"missing {source_column!r} column",
            "rows_before": int(len(frame)),
            "rows_after": int(len(frame)),
            "rows_removed": 0,
        }

    source_values = frame[source_column].astype("string").fillna("")
    if not source_values.isin(LICENSE_MAPPED_SOURCES).any():
        return frame.copy(), {
            "enabled": False,
            "reason": "no recognized license-mapped source values",
            "rows_before": int(len(frame)),
            "rows_after": int(len(frame)),
            "rows_removed": 0,
        }

    permitted = source_values.isin(PERMISSIVE_SOURCE_METADATA)
    filtered = frame.loc[permitted].copy()
    filtered["source_detail"] = filtered[source_column].astype("string").fillna("")
    filtered["source_dataset"] = filtered[source_column].map(
        lambda value: PERMISSIVE_SOURCE_METADATA[str(value)]["source_dataset"]
    )
    filtered["source_license"] = filtered[source_column].map(
        lambda value: PERMISSIVE_SOURCE_METADATA[str(value)]["source_license"]
    )
    filtered["upstream_url"] = filtered[source_column].map(
        lambda value: PERMISSIVE_SOURCE_METADATA[str(value)]["upstream_url"]
    )

    removed_sources = source_values.loc[~permitted].value_counts().to_dict()
    return filtered.reset_index(drop=True), {
        "enabled": True,
        "source_column": source_column,
        "rows_before": int(len(frame)),
        "rows_after": int(len(filtered)),
        "rows_removed": int(len(frame) - len(filtered)),
        "removed_sources": {str(key): int(value) for key, value in removed_sources.items()},
        "kept_sources": {
            str(key): int(value)
            for key, value in source_values.loc[permitted].value_counts().items()
        },
    }

"""Provider selection helpers for detector runtimes."""

from __future__ import annotations

from typing import Sequence


def parse_provider_list(value: str | None) -> list[str]:
    if not value:
        return []
    tokens = [item.strip() for item in value.replace(";", ",").split(",")]
    providers = [token for token in tokens if token]
    if providers:
        return providers
    return []


def resolve_providers(
    device: str,
    override: str | None,
    available: Sequence[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    override_list = parse_provider_list(override)
    if override_list:
        desired = override_list
    elif device == "cpu":
        desired = ["CPUExecutionProvider"]
    else:
        desired = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if available is None:
        return desired, [], desired

    available_set = {provider for provider in available}
    filtered = [provider for provider in desired if provider in available_set]
    missing = [provider for provider in desired if provider not in available_set]
    if not filtered:
        # Fallback to any available provider, prefer CPU if present.
        if "CPUExecutionProvider" in available_set:
            filtered = ["CPUExecutionProvider"]
        else:
            filtered = list(available)
    return filtered, missing, desired

from pickpresence.provider_utils import parse_provider_list, resolve_providers


def test_parse_provider_list():
    assert parse_provider_list("CUDAExecutionProvider,CPUExecutionProvider") == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert parse_provider_list("  CPUExecutionProvider ") == ["CPUExecutionProvider"]


def test_resolve_providers_fallback_to_cpu():
    providers, missing, desired = resolve_providers(
        device="auto",
        override=None,
        available=["CPUExecutionProvider"],
    )
    assert desired == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert "CUDAExecutionProvider" in missing
    assert providers == ["CPUExecutionProvider"]


def test_resolve_providers_override():
    providers, missing, desired = resolve_providers(
        device="auto",
        override="CPUExecutionProvider",
        available=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert desired == ["CPUExecutionProvider"]
    assert missing == []
    assert providers == ["CPUExecutionProvider"]

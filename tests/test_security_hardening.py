from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[1]


def test_ci_actions_are_sha_pinned_and_uv_not_latest():
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "version: \"latest\"" not in text
    assert "version: \"0.11.25\"" in text
    for action in ("actions/checkout", "actions/setup-python", "astral-sh/setup-uv"):
        assert re.search(rf"uses:\s*{re.escape(action)}@[0-9a-f]{{40}}", text)

from pathlib import Path
import json
import os
import re
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]


def test_graphify_skill_does_not_install_unpinned_packages():
    text = (REPO / ".github" / "skills" / "graphify" / "SKILL.md").read_text(encoding="utf-8")

    assert "break-system-packages" not in text
    assert not re.search(r"\bpip\s+install\b", text)
    assert "graphifyy==0.7.0" in (REPO / "pyproject.toml").read_text(encoding="utf-8")


def test_graphify_neo4j_password_not_embedded_in_inline_command():
    text = (REPO / ".github" / "skills" / "graphify" / "SKILL.md").read_text(encoding="utf-8")

    assert "password='NEO4J_PASSWORD'" not in text
    assert 'password="NEO4J_PASSWORD"' not in text
    assert "os.environ['NEO4J_PASSWORD']" in text


def test_ci_actions_are_sha_pinned_and_uv_not_latest():
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "version: \"latest\"" not in text
    assert "version: \"0.11.25\"" in text
    for action in ("actions/checkout", "actions/setup-python", "astral-sh/setup-uv"):
        assert re.search(rf"uses:\s*{re.escape(action)}@[0-9a-f]{{40}}", text)


def test_brainstorm_server_has_non_loopback_auth_hooks():
    server = REPO / ".github" / "skills" / "brainstorming" / "scripts" / "server.cjs"
    helper = REPO / ".github" / "skills" / "brainstorming" / "scripts" / "helper.js"
    start = REPO / ".github" / "skills" / "brainstorming" / "scripts" / "start-server.sh"

    server_text = server.read_text(encoding="utf-8")
    assert "BRAINSTORM_TOKEN" in server_text
    assert "requestAuthorized(req)" in server_text
    assert "originAllowed(req)" in server_text
    assert "401 Unauthorized" in server_text
    assert "token=" in helper.read_text(encoding="utf-8")
    assert "BRAINSTORM_TOKEN=" in start.read_text(encoding="utf-8")


def test_brainstorm_auth_predicate_with_node():
    if shutil.which("node") is None:
        pytest.skip("node is not available")

    script = REPO / ".github" / "skills" / "brainstorming" / "scripts" / "server.cjs"
    loopback_code = (
        f"const s = require({json.dumps(str(script))});"
        "if (!s.isLoopbackHost('127.0.0.1')) process.exit(1);"
        "if (!s.isLoopbackHost('localhost')) process.exit(1);"
        "if (s.isLoopbackHost('0.0.0.0')) process.exit(1);"
        "if (!s.requestAuthorized({url:'/', headers:{host:'localhost:1234'}})) process.exit(1);"
    )
    subprocess.run(["node", "-e", loopback_code], check=True)

    non_loopback_code = (
        f"const s = require({json.dumps(str(script))});"
        "const base = {headers:{host:'0.0.0.0:1234', origin:'http://0.0.0.0:1234'}};"
        "if (s.requestAuthorized({...base, url:'/'})) process.exit(1);"
        "if (s.requestAuthorized({...base, url:'/?token=wrong'})) process.exit(1);"
        "if (!s.requestAuthorized({...base, url:'/?token=secret'})) process.exit(1);"
    )
    subprocess.run(
        ["node", "-e", non_loopback_code],
        check=True,
        env={
            **os.environ,
            "BRAINSTORM_HOST": "0.0.0.0",
            "BRAINSTORM_URL_HOST": "0.0.0.0",
            "BRAINSTORM_TOKEN": "secret",
        },
    )

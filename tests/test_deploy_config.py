"""Static validation of the deploy/ pack (docker-compose.yaml, .env.example,
Dockerfiles, backup script) - deployment plumbing does not run kalos itself,
so this deliberately does not touch any `kalos.*` import: it just checks the
config files are well-formed and internally consistent with each other.

Does NOT attempt a real `docker build` (heavy, network-dependent, out of
scope per the deploy pack's brief - see deploy/RUNBOOK.md, "What was
verified"). Where the `docker` CLI is available, `test_compose_config_
resolves_via_docker_cli` additionally shells out to `docker compose config`
for a real parse by Compose itself, not just PyYAML; it skips cleanly where
`docker` is absent so this file stays runnable without Docker installed.
"""
from __future__ import annotations

import re
import subprocess
import shutil
from pathlib import Path

import pytest
import yaml

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"
COMPOSE_PATH = DEPLOY_DIR / "docker-compose.yaml"
ENV_EXAMPLE_PATH = DEPLOY_DIR / ".env.example"

# Every env var this deploy pack treats as REQUIRED - i.e. the stack starts
# wrong (or, for the security three, refuses to start at all per
# kalos/portal/config.py's assert_safe_exposure) without a real value here.
REQUIRED_ENV_KEYS = [
    "KALOS_AUTH_TOKENS",
    "KALOS_ANON_SALT",
    "KALOS_CORS_ORIGINS",
    "NEXT_PUBLIC_API_URL",
]


def _load_compose() -> dict:
    with COMPOSE_PATH.open() as f:
        return yaml.safe_load(f)


def test_compose_file_parses_as_yaml():
    doc = _load_compose()
    assert isinstance(doc, dict)
    assert "services" in doc


def test_compose_declares_expected_services():
    doc = _load_compose()
    services = doc["services"]
    assert set(services) == {"engine", "web", "backup"}


def test_compose_publishes_only_the_web_front_door():
    """Exactly ONE host port: `web`. Wave 2 removed the engine's published
    port - the browser calls same-origin /api/engine and kalos-web's SERVER
    proxies to the engine over the compose-internal network with the bearer
    token attached, so publishing the engine would reopen the unauthenticated
    surface the proxy exists to close. `backup` has nothing listening and
    stays unreachable, per the reference topology's "only the edge services
    publish a port" pattern (deployment-reference.md)."""
    doc = _load_compose()
    services = doc["services"]
    assert "ports" not in services["engine"]
    assert "ports" in services["web"]
    assert "ports" not in services["backup"]


def test_compose_web_carries_the_proxy_env():
    """The proxy needs both server-side vars; a compose without them ships a
    web container whose every engine call 401s or 502s."""
    doc = _load_compose()
    env = doc["services"]["web"].get("environment", [])
    joined = " ".join(env) if isinstance(env, list) else " ".join(f"{k}={v}" for k, v in env.items())
    assert "KALOS_ENGINE_URL" in joined
    assert "KALOS_ENGINE_TOKEN" in joined


def test_compose_engine_has_no_replicas_override():
    """kalos/store/sqlite_store.py and kalos/runner/singleton.py assume a
    single writer; this compose file must not silently scale the engine."""
    doc = _load_compose()
    engine = doc["services"]["engine"]
    deploy = engine.get("deploy", {})
    assert deploy.get("replicas", 1) == 1


def test_compose_engine_binds_state_dir_to_engine_home_and_backup_reads_it_readonly():
    """kalos/store/sqlite_store.py's SqliteStore and
    kalos/runner/singleton.py's DEFAULT_LOCK_PATH both hardcode
    Path.home()/".kalos" and ignore KALOS_STATE_DIR - so the volume mount
    target must be the engine container's actual $HOME/.kalos
    (/home/kalos/.kalos per Dockerfile.engine), not an arbitrary path, or
    experiments.db and runner.lock silently land outside the mounted volume
    (and outside the backup sidecar's reach)."""
    doc = _load_compose()
    engine_env = doc["services"]["engine"]["environment"]
    assert "KALOS_STATE_DIR=/home/kalos/.kalos" in engine_env

    engine_volumes = doc["services"]["engine"]["volumes"]
    assert "kalos-state:/home/kalos/.kalos" in engine_volumes

    backup_volumes = doc["services"]["backup"]["volumes"]
    assert any(
        v.startswith("kalos-state:/home/kalos/.kalos") and v.endswith(":ro")
        for v in backup_volumes
    ), f"expected a read-only kalos-state mount in backup volumes, got {backup_volumes!r}"


def test_compose_backup_target_is_outside_the_named_volume_set():
    """Backups must survive a `docker compose down -v` against the primary
    `kalos-state` volume, so the backup destination must be a bind mount to
    a host path, not another entry in the same named-volume set."""
    doc = _load_compose()
    backup_volumes = doc["services"]["backup"]["volumes"]
    named_volumes = set(doc.get("volumes", {}) or {})
    backup_dest = next(v for v in backup_volumes if v.endswith("/backups"))
    host_path = backup_dest.split(":")[0]
    assert host_path not in named_volumes
    assert host_path.startswith("./") or host_path.startswith("/")


def test_compose_restart_policies_match_reference_pattern():
    doc = _load_compose()
    services = doc["services"]
    assert services["engine"]["restart"] == "unless-stopped"
    assert services["web"]["restart"] == "unless-stopped"
    assert services["backup"]["restart"] == "always"


def test_compose_engine_auth_and_salt_env_are_wired_from_env_file():
    """The engine binds 0.0.0.0 in this stack (KALOS_HOST=0.0.0.0), which
    kalos/portal/config.py's assert_safe_exposure refuses to serve without
    auth configured and KALOS_ANON_SALT set. This asserts the compose file
    actually passes both through from the host `.env` rather than, say,
    leaving them unset and relying on KALOS_ALLOW_OPEN_ACCESS by accident."""
    doc = _load_compose()
    engine_env = doc["services"]["engine"]["environment"]
    assert "KALOS_HOST=0.0.0.0" in engine_env
    assert any(e.startswith("KALOS_AUTH_TOKENS=") for e in engine_env)
    assert any(e.startswith("KALOS_ANON_SALT=") for e in engine_env)
    assert any(e.startswith("KALOS_CORS_ORIGINS=") for e in engine_env)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")
def test_compose_config_resolves_via_docker_cli(tmp_path):
    """A real parse by `docker compose config`, not just PyYAML - catches
    anything PyYAML would accept but Compose's own schema would not (e.g. a
    field compose doesn't recognize). Uses a throwaway filled-in copy of
    .env.example so this doesn't depend on a real deploy/.env existing."""
    env_content = ENV_EXAMPLE_PATH.read_text()
    placeholders = {
        "REPLACE_ME_WITH_A_REAL_SHA256_HASH": "0" * 64,
        "REPLACE_ME_WITH_A_RANDOM_SECRET": "test-salt-not-for-real-use",
        "REPLACE_ME_WITH_YOUR_WEB_HOST": "localhost",
        "REPLACE_ME_WITH_YOUR_HOST": "localhost",
    }
    for needle, replacement in placeholders.items():
        env_content = env_content.replace(needle, replacement)
    env_path = tmp_path / ".env"
    env_path.write_text(env_content)

    result = subprocess.run(
        ["docker", "compose", "--env-file", str(env_path), "-f", str(COMPOSE_PATH), "config", "--quiet"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 and "docker daemon" in (result.stderr or "").lower():
        pytest.skip(f"docker daemon not reachable in this environment: {result.stderr.strip()}")
    assert result.returncode == 0, result.stderr


def test_env_example_documents_every_required_key():
    text = ENV_EXAMPLE_PATH.read_text()
    for key in REQUIRED_ENV_KEYS:
        # Required keys are live assignments (`KEY=value`), not commented
        # out - a commented-out REQUIRED key would silently ship an unset
        # config that assert_safe_exposure then refuses at engine startup.
        pattern = re.compile(rf"^{re.escape(key)}=", re.MULTILINE)
        assert pattern.search(text), f"{key} must appear as a live (uncommented) assignment in .env.example"


def test_env_example_required_keys_are_not_left_as_bare_placeholders_only():
    """Sanity check on the placeholder CONTENT itself: each required key's
    example value must contain a REPLACE_ME marker (so a copy-paste deploy
    fails loudly / obviously, not with a silently-accepted example value)."""
    text = ENV_EXAMPLE_PATH.read_text()
    for key in REQUIRED_ENV_KEYS:
        match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.MULTILINE)
        assert match, f"{key} not found"
        assert "REPLACE_ME" in match.group(1), (
            f"{key}'s example value should contain REPLACE_ME so it's obvious "
            "it needs a real value, not a value that looks plausible as-is"
        )


# Known-shape secret prefixes/patterns that must never appear as a literal in
# a committed deploy/ file. Deliberately narrow (specific vendor prefixes,
# not "any 32+ char hex string") so this doesn't false-positive on the
# placeholder SHA-256 hash examples or the deadbeef... test fixture in this
# file, which are documentation, not secrets.
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),   # Anthropic live key
    re.compile(r"sk-[A-Za-z0-9]{20,}"),          # OpenAI-style live key
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),         # GitHub PAT
    re.compile(r"AKIA[0-9A-Z]{16}"),             # AWS access key id
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack token
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),       # Google API key
]

# Lines that are documentation ABOUT the shape of a secret (e.g. an example
# `.env` var name and a REPLACE_ME placeholder) are fine; this only exists to
# catch an actual live-looking literal accidentally committed.
_ALLOWED_SUBSTRINGS = ("REPLACE_ME", "deadbeef" * 8, "0" * 64)


def _iter_deploy_files():
    for path in DEPLOY_DIR.rglob("*"):
        if path.is_file():
            yield path


def test_no_secret_looking_literal_in_any_deploy_file():
    offenders = []
    for path in _iter_deploy_files():
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in _SECRET_PATTERNS:
            for m in pattern.finditer(text):
                token = m.group(0)
                if any(allowed in token for allowed in _ALLOWED_SUBSTRINGS):
                    continue
                offenders.append((path, token))
    assert not offenders, f"secret-looking literals found: {offenders}"


def test_dockerfiles_run_as_non_root_user():
    for name in ("Dockerfile.engine", "Dockerfile.web", "backup/Dockerfile"):
        text = (DEPLOY_DIR / name).read_text()
        user_lines = [line for line in text.splitlines() if line.strip().startswith("USER ")]
        assert user_lines, f"{name} never switches to a non-root USER"
        assert user_lines[-1].strip() != "USER root", f"{name} ends as root"


def test_dockerfiles_declare_a_healthcheck():
    for name in ("Dockerfile.engine", "Dockerfile.web"):
        text = (DEPLOY_DIR / name).read_text()
        assert "HEALTHCHECK" in text, f"{name} has no HEALTHCHECK"


def test_dockerignore_keeps_engine_inputs_and_drops_heavy_or_secret_paths():
    """The engine builds from the repo root, so `.dockerignore` decides what is
    sent to the daemon: it must keep every path Dockerfile.engine COPYs, and
    must drop the dev venv (GBs), the git dir, and any real `.env`."""
    ignore = (DEPLOY_DIR.parent / ".dockerignore").read_text().splitlines()
    patterns = {line.strip() for line in ignore if line.strip() and not line.startswith("#")}
    for must_drop in (".venv/", ".git/", ".env"):
        assert must_drop in patterns, f".dockerignore does not exclude {must_drop}"
    engine = (DEPLOY_DIR / "Dockerfile.engine").read_text()
    copied = []
    for line in engine.splitlines():
        parts = line.split()
        if parts[:1] == ["COPY"] and not any(p.startswith("--from") for p in parts):
            copied.extend(parts[1:-1])
    assert copied, "found no COPY sources in Dockerfile.engine"
    for src in copied:
        name = src.rstrip("/")
        assert name not in {p.rstrip("/") for p in patterns}, f".dockerignore drops {src}, which Dockerfile.engine COPYs"


def test_backup_script_is_syntactically_valid_posix_sh():
    script = DEPLOY_DIR / "backup" / "backup.sh"
    result = subprocess.run(["sh", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_backup_script_state_dir_default_matches_engine_home():
    """The backup sidecar's default STATE_DIR must match Dockerfile.engine's
    KALOS_STATE_DIR - a drift between the two would make the sidecar back up
    an empty directory."""
    script_text = (DEPLOY_DIR / "backup" / "backup.sh").read_text()
    dockerfile_text = (DEPLOY_DIR / "Dockerfile.engine").read_text()
    assert "/home/kalos/.kalos" in script_text
    assert "KALOS_STATE_DIR=/home/kalos/.kalos" in dockerfile_text

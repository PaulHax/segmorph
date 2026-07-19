"""Shared fixture-root resolution for the Python oracle generators.

`npm run fixtures` writes the committed corpus under test/fixtures. The oracle
test tier instead runs the generators with SEGMORPH_FIXTURES_DIR pointing at
test/generated, so specs compare against goldens computed live by the pinned
oracles rather than committed files that can go stale. Every generator resolves
its output root — and any golden it reads back as input — through here, so the
redirect is all-or-nothing and a live run can never mix the two corpora.
"""

import json
import os
import pathlib


def fixtures_root(root):
    """Resolve the fixture corpus root, honoring the SEGMORPH_FIXTURES_DIR redirect."""
    override = os.environ.get("SEGMORPH_FIXTURES_DIR")
    return pathlib.Path(override) if override else pathlib.Path(root) / "test" / "fixtures"


def read_manifest(path):
    """Read a fixture manifest, tolerating the empty corpus a live run starts from."""
    if not pathlib.Path(path).exists():
        return {"schemaVersion": 1, "fixtures": []}
    manifest = json.loads(pathlib.Path(path).read_text())
    assert manifest["schemaVersion"] == 1 and isinstance(manifest["fixtures"], list)
    return manifest


def write_manifest(path, manifest):
    pathlib.Path(path).write_text(json.dumps(manifest, indent=2) + "\n")

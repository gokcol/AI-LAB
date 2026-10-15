"""Validate public release notes; prepend a new version or a numbered revision."""

import argparse
import datetime as dt
import json
from pathlib import Path
import re


def release_order(entry):
    return (tuple(int(part) for part in entry["version"].split(".")), entry.get("revision", 1))


def validate(path, previous=None, *, redact_versions=()):
    releases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(releases, list) or not releases:
        raise ValueError("Release notes must be a nonempty list")
    versions, dates = [], []
    for entry in releases:
        version = entry.get("version", "")
        if not re.fullmatch(r"\d+(?:\.\d+)+", version):
            raise ValueError("Each release needs a numeric version, e.g. 1.03")
        revision = entry.get("revision", 1)
        if type(revision) is not int or revision < 1:
            raise ValueError("A release revision must be a positive integer")
        versions.append(release_order(entry))
        dates.append(dt.date.fromisoformat(entry["date"]))
        if not entry.get("title", "").strip() or not isinstance(entry.get("changes"), list) or not entry["changes"]:
            raise ValueError("Each release needs a title and user-facing changes")
        if any(not isinstance(change, str) or not change.strip() for change in entry["changes"]):
            raise ValueError("Release changes cannot be empty")
    if any(a <= b for a, b in zip(versions, versions[1:])):
        raise ValueError("Version/revision pairs must be unique and newest first")
    if any(a < b for a, b in zip(dates, dates[1:])) or dates[0] > dt.date.today():
        raise ValueError("Release dates must be newest first and not in the future")
    if previous is not None and Path(previous).exists():
        old = validate(previous)
        # Only use this explicit exception when the owner requests removal of a
        # published entry. Every other historical entry must remain unchanged.
        redacted = set(redact_versions)
        if not redacted <= {entry["version"] for entry in old}:
            raise ValueError("Redacted versions must exist in the published history")
        if redacted & {entry["version"] for entry in releases}:
            raise ValueError("Redacted versions must be removed")
        kept = [entry for entry in old if entry["version"] not in redacted]
        old_latest = release_order(old[0])
        if (versions[0] <= old_latest or len(releases) <= len(kept)
                or (kept and releases[-len(kept):] != kept)):
            raise ValueError("Prepend a new release entry; preserve published history")
    elif redact_versions:
        raise ValueError("Redaction requires a previous release file")
    return releases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--redact-version", action="append", default=[],
                        help="Owner-requested removal of this historical entry; repeat for multiple versions")
    args = parser.parse_args()
    try:
        releases = validate(Path(__file__).resolve().parents[1] / "gui/content/releases.json",
                            args.previous, redact_versions=args.redact_version)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"Release notes check failed: {error}\n")
    print(f"Release notes ready: v{releases[0]['version']} (revision {releases[0].get('revision', 1)})")

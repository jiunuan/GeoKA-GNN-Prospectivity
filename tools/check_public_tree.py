"""Fail closed on unexpected files in a source-only publication folder.

This checks the physical tree, not just .gitignore. Run on the clean release
folder; local experiments must be removed from that copy before publication.
"""
import argparse
import hashlib
from pathlib import Path
import re
import sys

ROOT_FILES = {"README.md", "README.zh-CN.md", "pyproject.toml", ".gitignore", ".gitattributes", "PUBLICATION_FILES.txt"}
PATTERNS = [
    ("absolute Windows path", re.compile(r"(?i)[a-z]:[\\/]")),
    ("private-key material", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
    ("credential assignment", re.compile(r"(?i)(?:api_key|access_token|password)\s*[:=]\s*['\"][^'\"]{6,}")),
    ("local user directory", re.compile(r"(?i)/(?:home|Users)/[a-z0-9_.-]+/")),
]


def audit(root):
    failures, files = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            failures.append(f"Symlink forbidden: {relative.as_posix()}")
            continue
        if path.is_dir():
            continue
        name = relative.as_posix()
        allowed = (name in ROOT_FILES or name == "configs/default.yaml"
                   or relative.parts[0] == "src" and path.suffix == ".py"
                   or relative.parts[0] == "tests" and path.suffix == ".py"
                   or relative.parts[0] == "tools" and path.suffix == ".py"
                   or relative.parts[0] == "docs" and path.suffix == ".md")
        if not allowed:
            failures.append(f"Unexpected/non-source file: {name}")
            continue
        if path.stat().st_size > 150000:
            failures.append(f"Unexpectedly large source: {name}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            failures.append(f"Not UTF-8 text: {name}")
            continue
        for reason, expression in PATTERNS:
            if expression.search(text):
                failures.append(f"{reason}: {name}")
        files.append(name)
    return files, failures


def write_manifest(root, files):
    lines = ["Source-only publication inventory (this file excludes itself).", "SHA256  relative_path", ""]
    for name in sorted(set(files) - {"PUBLICATION_FILES.txt"}):
        lines.append(hashlib.sha256((root / name).read_bytes()).hexdigest() + "  " + name)
    (root / "PUBLICATION_FILES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_manifest(root, files):
    path = root / "PUBLICATION_FILES.txt"
    if not path.exists():
        return ["Publication inventory missing"]
    entries, failures = {}, []
    for line in path.read_text(encoding="utf-8").splitlines()[3:]:
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            failures.append("Invalid publication inventory entry")
            continue
        digest, name = match.groups()
        if name in entries:
            failures.append(f"Duplicate inventory entry: {name}")
        entries[name] = digest
    expected = set(files) - {"PUBLICATION_FILES.txt"}
    failures.extend(f"Inventory entry missing: {name}" for name in sorted(expected - set(entries)))
    failures.extend(f"Unexpected inventory entry: {name}" for name in sorted(set(entries) - expected))
    # Read only names found by the validated source-tree scan.
    for name in sorted(expected & set(entries)):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != entries[name]:
            failures.append(f"Inventory SHA256 mismatch: {name}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-manifest", action="store_true",
                        help="Refresh SHA256 inventory after a successful source-only scan")
    args = parser.parse_args()
    root = args.root.resolve()
    files, failures = audit(root)
    if not failures:
        if args.write_manifest:
            write_manifest(root, files)
        failures.extend(verify_manifest(root, files))
    if failures:
        print("PUBLIC SOURCE CHECK FAILED\n" + "\n".join(failures))
        sys.exit(1)
    print(f"PUBLIC SOURCE CHECK PASSED: {len(files)} UTF-8 source/config/documentation files; no data or weights.")


if __name__ == "__main__":
    main()

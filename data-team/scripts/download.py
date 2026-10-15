"""Download the TORSO-21 dataset into data-team/datasets/TORSO-21/.

TORSO-21 is hosted as a plain directory listing at
https://data.bit-bots.de/TORSO-21/ (the official repo's
`scripts/download_dataset.py` pulls from the same place). This script
walks that listing and mirrors the parts we ask for, so we don't need the
upstream repo's Poetry environment just to fetch data.

Examples:
    # Real-world test split only (smallest; good for a first look)
    python data-team/scripts/download.py --subset reality --split test

    # Everything real-world (train + test)
    python data-team/scripts/download.py --subset reality

    # See what would be downloaded without downloading it
    python data-team/scripts/download.py --subset reality --dry-run

Zip archives are extracted next to where they were downloaded unless
--no-extract is given. Files that already exist with the same size are
skipped, so the script is safe to re-run after an interrupted download.

See data-team/README.md ("Dataset schema") for what you get.
"""

import argparse
import sys
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

BASE_URL = "https://data.bit-bots.de/TORSO-21/"
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
CHUNK = 1 << 20


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def list_dir(url: str) -> list:
    """Return absolute child URLs of an HTML directory listing."""
    with urllib.request.urlopen(url) as resp:
        parser = _LinkParser()
        parser.feed(resp.read().decode("utf-8", errors="replace"))
    children = []
    for href in parser.links:
        if href.startswith(("?", "#", "/", "..")) or "://" in href:
            continue  # sort links, parent dir, absolute/external links
        children.append(urllib.parse.urljoin(url, href))
    return sorted(set(children))


def walk(url: str):
    """Yield every file URL below a directory URL."""
    for child in list_dir(url):
        if child.endswith("/"):
            yield from walk(child)
        else:
            yield child


def wanted(rel_path: str, subset: str, split: str) -> bool:
    parts = rel_path.split("/")
    other_subsets = {"reality", "simulation"} - {subset}
    if subset != "all" and parts[0] != subset:
        # Top-level files (README, LICENSE, ...) are kept, unless they are
        # an archive of the other subset (e.g. simulation.zip).
        return len(parts) == 1 and not parts[0].startswith(tuple(other_subsets))
    if split != "all" and len(parts) > 2 and parts[1] in ("train", "test"):
        return parts[1] == split
    return True


def remote_size(url: str):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except OSError:  # URLError/HTTPError are OSErrors
        return None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(url)
    if dest.exists() and size is not None and dest.stat().st_size == size:
        print(f"  skip (already have) {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        done = 0
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if size:
                print(f"\r  {dest.name}: {done * 100 // size:3d}%", end="", flush=True)
    print(f"\r  done {dest.name}" + " " * 10)
    tmp.replace(dest)


def extract(archive: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        root = archive.parent.resolve()
        for member in zf.namelist():
            # Refuse path traversal ("../") in archive member names.
            if not (archive.parent / member).resolve().is_relative_to(root):
                raise RuntimeError(f"Unsafe path in {archive}: {member}")
        print(f"  extracting {archive.name}")
        zf.extractall(archive.parent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dest", type=Path, default=DATASETS_DIR,
                        help="Directory to download into (a TORSO-21/ folder is created inside).")
    parser.add_argument("--subset", choices=["reality", "simulation", "all"], default="reality",
                        help="Real-world images, Webots simulation images, or both. Default: reality.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all",
                        help="Which split to fetch. Default: all.")
    parser.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="List files, don't download.")
    parser.add_argument("--no-extract", action="store_true", help="Keep .zip files packed.")
    args = parser.parse_args()

    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
    out_root = args.dest / "TORSO-21"

    try:
        files = [u for u in walk(base) if wanted(u[len(base):], args.subset, args.split)]
    except OSError as exc:  # network errors, 403s, etc.
        sys.exit(f"Could not list {base}: {exc}\n"
                 "Check your connection, or download manually from that URL.")
    if not files:
        sys.exit(f"Nothing matched --subset {args.subset} --split {args.split} at {base}")

    print(f"{len(files)} file(s) -> {out_root}")
    for url in files:
        rel = urllib.parse.unquote(url[len(base):])
        dest = out_root / rel
        print(rel)
        if args.dry_run:
            continue
        download(url, dest)
        if dest.suffix == ".zip" and not args.no_extract:
            extract(dest)


if __name__ == "__main__":
    main()

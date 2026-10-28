"""Download public AVSBench media archives with pinned revision and SHA256.

The source is an anonymous public mirror, not the original application endpoint.
The original AVSBench dataset license still applies. No gated endpoint is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import zipfile

import requests


REVISION = "efbd6ec934706877e70577fb6a1997a7ffedfd67"
BASE = f"https://huggingface.co/datasets/UnFaZeD07/AVSBench/resolve/{REVISION}"
ASSETS = {
    "s4": ("s4_data.tar.gz", 7356668079, "cc6358a10f4cc1944fbe6dd77165ac8d39d864722fc2735e1567aef9c972194b"),
    "ms3": ("Multi-sources.zip", 656978149, "4109078d51e70e0dd12572418112e7af31cfa5d3fe143c3316c205e33152560f"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(kind, root, workers=8):
    name, size, expected_hash = ASSETS[kind]
    directory = root / "_downloads"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.is_file():
        if target.stat().st_size != size or sha256(target) != expected_hash:
            raise ValueError("existing archive differs from pinned asset; not overwriting")
        return target
    parts = directory / (name + ".parts")
    parts.mkdir(exist_ok=True)
    step = 32 * 1024 * 1024
    ranges = [(start, min(size - 1, start + step - 1)) for start in range(0, size, step)]

    def fetch(bounds):
        start, end = bounds
        path = parts / f"{start}-{end}"
        count = end - start + 1
        if path.exists() and path.stat().st_size == count:
            return count
        temporary = path.with_suffix(".partial")
        for attempt in range(5):
            try:
                with requests.get(f"{BASE}/{name}?download=true&range_start={start}",
                                  headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("server did not honor exact HTTP range")
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                if temporary.stat().st_size != count:
                    raise ValueError("incomplete download range")
                temporary.replace(path)
                return count
            except (requests.RequestException, ValueError):
                if attempt == 4:
                    raise RuntimeError(f"download failed after retries: {name} bytes {start}-{end}") from None
                time.sleep(attempt + 1)

    complete, started = 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(fetch, bounds) for bounds in ranges]):
            complete += future.result()
            print(f"{name}: verified ranges {complete}/{size} ({complete/size:.1%}), elapsed={time.monotonic()-started:.0f}s", flush=True)
    partial = target.with_suffix(target.suffix + ".partial")
    with partial.open("wb") as output:
        for start, end in ranges:
            with (parts / f"{start}-{end}").open("rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    actual_hash = sha256(partial)
    if partial.stat().st_size != size or actual_hash != expected_hash:
        raise ValueError("whole-archive SHA256 mismatch; asset not accepted")
    partial.replace(target)
    record = {"mirror": "UnFaZeD07/AVSBench", "mirror_revision": REVISION, "url": f"{BASE}/{name}",
              "bytes": size, "sha256": actual_hash, "license": "original AVSBench CC BY-NC 4.0; not the repository code license",
              "verification": "full file matches public HF LFS SHA256; official metadata/masks verified separately"}
    target.with_suffix(target.suffix + ".source.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(parts)
    print(f"verified archive {target}", flush=True)
    return target


def safe_member_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe archive member path")
    return path


def extract_asset(kind, archive, root):
    destination = root / ("Single-source" if kind == "s4" else "Multi-sources")
    extracted, skipped = 0, 0

    def output_for(name):
        nonlocal skipped
        path = safe_member_path(name)
        parts = list(path.parts)
        if any(p.startswith(".") or p == "__MACOSX" for p in parts):
            skipped += 1
            return None
        if parts and parts[0] in ("Single-source", "Multi-sources"):
            parts = parts[1:]
        if not parts or not any(p in ("raw_videos", "visual_frames", "audio_wav", "gt_masks") for p in parts):
            skipped += 1
            return None
        if "audio_log_mel" in parts:
            skipped += 1
            return None
        return destination.joinpath(*parts)

    def save(source, target):
        nonlocal extracted
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".extracting")
        with temporary.open("wb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        if target.exists():
            if sha256(target) != sha256(temporary):
                temporary.unlink()
                raise ValueError(f"existing file differs from archive: {target}")
            temporary.unlink()
        else:
            temporary.replace(target)
        extracted += 1
        if extracted % 5000 == 0:
            print(f"{kind}: extracted/verified {extracted} media/mask files", flush=True)

    if kind == "s4":
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                if member.issym() or member.islnk():
                    raise ValueError("archive links are not accepted")
                if member.isfile():
                    target = output_for(member.name)
                    if target is not None:
                        with stream.extractfile(member) as source:
                            save(source, target)
    else:
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("archive links are not accepted")
                if not member.is_dir():
                    target = output_for(member.filename)
                    if target is not None:
                        with stream.open(member) as source:
                            save(source, target)
    record = {"asset": str(archive), "destination": str(destination), "extracted_or_verified_files": extracted,
              "skipped_files": skipped, "audio_log_mel_pickles_used": False}
    (root / f"{kind}.media_extraction.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=("s4", "ms3", "all"), default="all")
    parser.add_argument("--root", type=Path, default=Path("/nvme_data/pkt/streaming/dataset/avsbench"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    for kind in (("s4", "ms3") if args.asset == "all" else (args.asset,)):
        archive = download_asset(kind, args.root, args.workers)
        if args.extract:
            extract_asset(kind, archive, args.root)


if __name__ == "__main__":
    main()

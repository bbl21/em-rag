"""Controlled acquisition of public reference sources.

The module deliberately has no product-specific catalogue embedded in it.  A
caller supplies a small JSON catalogue and an explicit host allowlist, making
the network boundary reviewable and suitable for a bootstrap CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


class SourceAcquisitionError(ValueError):
    """Raised when a catalogue entry or downloaded source violates policy."""


_SAFE_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    url: str
    filename: str
    sha256: str
    max_bytes: int
    acquisition_mode: str = "download"
    file_type: str = "file"


@dataclass(frozen=True)
class AcquisitionResult:
    source_id: str
    status: str
    path: Path | None
    receipt_path: Path | None
    message: str


def _validate_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise SourceAcquisitionError("source URL must use HTTPS and include a host")
    if parsed.username or parsed.password:
        raise SourceAcquisitionError("source URL must not contain credentials")
    if parsed.hostname.lower() not in allowed_hosts:
        raise SourceAcquisitionError(f"source host is not allowlisted: {parsed.hostname}")


def _validate_source_paths(source_id: str, filename: str) -> None:
    if source_id in {".", ".."} or not _SAFE_SOURCE_ID.fullmatch(source_id):
        raise SourceAcquisitionError("source_id must contain only letters, digits, dots, underscores, or hyphens")
    if not filename or Path(filename).name != filename:
        raise SourceAcquisitionError("filename must be a non-empty basename without a path")


def _path_inside(root: Path, name: str) -> Path:
    root = root.resolve()
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SourceAcquisitionError("source output path escapes destination_root") from error
    return candidate


def _source_from_mapping(entry: Mapping[str, Any], allowed_hosts: set[str]) -> SourceSpec:
    try:
        source = SourceSpec(
            source_id=str(entry["source_id"]),
            url=str(entry.get("url", "")),
            filename=str(entry["filename"]),
            sha256=str(entry.get("sha256", "")).lower(),
            max_bytes=int(entry["max_bytes"]),
            acquisition_mode=str(entry.get("acquisition_mode", "download")),
            file_type=str(entry.get("file_type", "file")).lower(),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SourceAcquisitionError("source entries require source_id, filename, and positive max_bytes") from error
    _validate_source_paths(source.source_id, source.filename)
    if source.max_bytes <= 0:
        raise SourceAcquisitionError("max_bytes must be positive")
    if source.acquisition_mode not in {"download", "manual_download"}:
        raise SourceAcquisitionError("acquisition_mode must be download or manual_download")
    if source.file_type not in {"file", "pdf", "zip", "tar"}:
        raise SourceAcquisitionError("file_type must be file, pdf, zip, or tar")
    _validate_url(source.url, allowed_hosts)
    if source.acquisition_mode == "download":
        if len(source.sha256) != 64 or any(char not in "0123456789abcdef" for char in source.sha256):
            raise SourceAcquisitionError("download sources require a lowercase SHA-256 value")
    return source


def load_source_catalog(path: Path, allowed_hosts: set[str] | list[str] | tuple[str, ...]) -> list[SourceSpec]:
    """Load and validate a JSON catalogue (a list or ``{\"sources\": [...]}``)."""
    hosts = {host.lower() for host in allowed_hosts}
    if not hosts:
        raise SourceAcquisitionError("an explicit host allowlist is required")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceAcquisitionError(f"cannot read source catalogue: {path}") from error
    entries = document.get("sources") if isinstance(document, dict) else document
    if not isinstance(entries, list) or not entries:
        raise SourceAcquisitionError("catalogue must contain a non-empty sources list")
    sources = [_source_from_mapping(entry, hosts) for entry in entries if isinstance(entry, Mapping)]
    if len(sources) != len(entries) or len({source.source_id for source in sources}) != len(sources):
        raise SourceAcquisitionError("catalogue entries must be mappings with unique source_id values")
    return sources


def _safe_member(name: str) -> bool:
    value = PurePosixPath(name)
    return bool(name) and not value.is_absolute() and ".." not in value.parts


def _extract_archive(archive: Path, target: Path, file_type: str) -> None:
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        if file_type == "zip":
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                if any(not _safe_member(member.filename) or ((member.external_attr >> 16) & 0o170000) == 0o120000 for member in members):
                    raise SourceAcquisitionError("archive contains unsafe path or symlink")
                bundle.extractall(staging, members)
        elif file_type == "tar":
            with tarfile.open(archive) as bundle:
                members = bundle.getmembers()
                if any(not _safe_member(member.name) or member.issym() or member.islnk() for member in members):
                    raise SourceAcquisitionError("archive contains unsafe path or link")
                bundle.extractall(staging, members, filter="data")
        else:
            return
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def acquire_source(
    source: SourceSpec,
    destination_root: Path,
    allowed_hosts: set[str] | list[str] | tuple[str, ...],
    *,
    opener: Callable[..., Any] = urlopen,
    chunk_size: int = 64 * 1024,
) -> AcquisitionResult:
    """Acquire one validated source and write a UTF-8 JSON receipt.

    ``opener`` is injectable so callers can test without a network connection.
    """
    hosts = {host.lower() for host in allowed_hosts}
    _validate_source_paths(source.source_id, source.filename)
    _validate_url(source.url, hosts)
    if source.acquisition_mode == "manual_download":
        return AcquisitionResult(source.source_id, "manual_required", None, None,
                                 f"Provide {source.filename} manually for source {source.source_id}; automatic download is disabled.")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_root = destination_root.resolve()
    final_path = _path_inside(destination_root, source.filename)
    partial_path = final_path.with_name(final_path.name + ".partial")
    digest = hashlib.sha256()
    total = 0
    try:
        request = Request(source.url, headers={"User-Agent": "em-rag-source-bootstrap/1"})
        response = opener(request, timeout=30)
        with response, partial_path.open("wb") as output:
            while block := response.read(chunk_size):
                total += len(block)
                if total > source.max_bytes:
                    raise SourceAcquisitionError(f"download exceeds max_bytes ({source.max_bytes})")
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != source.sha256:
            raise SourceAcquisitionError("download SHA-256 does not match catalogue")
        if source.file_type == "pdf" and not partial_path.read_bytes()[:5].startswith(b"%PDF-"):
            raise SourceAcquisitionError("downloaded file is not a PDF")
        partial_path.replace(final_path)
        extracted = None
        if source.file_type in {"zip", "tar"}:
            extracted = _path_inside(destination_root, source.source_id)
            _extract_archive(final_path, extracted, source.file_type)
        receipt_path = _path_inside(destination_root, f"{source.source_id}.receipt.json")
        receipt = {**asdict(source), "sha256_observed": digest.hexdigest(), "bytes": total,
                   "path": str(final_path), "extracted_path": str(extracted) if extracted else None}
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return AcquisitionResult(source.source_id, "acquired", final_path, receipt_path, "source acquired and verified")
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

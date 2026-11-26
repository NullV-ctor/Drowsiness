"""Import the official MRL ZIP without extracting arbitrary archive paths.

Usage: python prepare_mrl.py data/raw/mrlEyes_2018_01.zip
"""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://mrl.cs.vsb.cz/eyedataset.html"
ARCHIVE_URL = "https://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
MRL_NAME = re.compile(
    r"^(s\d+)_(\d+)_([01])_([01])_([01])_([012])_([01])_(\d+)\.png$",
    re.IGNORECASE,
)


def parse_mrl_name(name):
    match = MRL_NAME.fullmatch(Path(name).name)
    if not match:
        raise ValueError(f"Not an MRL image filename: {name}")
    (
        subject,
        image_id,
        gender,
        glasses,
        state,
        reflections,
        lighting,
        sensor,
    ) = match.groups()
    # Official source: 0=closed, 1=open. Project: 0=open, 1=closed.
    return {
        "subject_id": subject.lower(),
        "image_id": image_id,
        "label": 1 - int(state),
        "source_eye_state": int(state),
        "gender": int(gender),
        "glasses": int(glasses),
        "reflections": int(reflections),
        "lighting": int(lighting),
        "sensor": sensor,
    }


def import_mrl(archive, output="data/eyes"):
    archive, output = Path(archive), Path(output)
    rows = []
    with zipfile.ZipFile(archive) as source:
        for member in sorted(source.infolist(), key=lambda item: item.filename):
            if member.is_dir() or not member.filename.lower().endswith(".png"):
                continue
            metadata = parse_mrl_name(member.filename)
            name = Path(member.filename).name
            state = "closed" if metadata["label"] else "open"
            relative = Path(state) / metadata["subject_id"] / name
            destination = output / relative
            content = source.read(member)
            digest = hashlib.sha256(content).hexdigest()
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise ValueError(
                        f"Existing file conflicts with archive: {destination}"
                    )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            rows.append(
                {
                    "relative_path": relative.as_posix(),
                    "file_sha256": digest,
                    **metadata,
                }
            )
    if not rows:
        raise ValueError("No MRL eye images found in archive.")
    frame = pd.DataFrame(rows)
    if frame.relative_path.duplicated().any():
        raise ValueError("Archive contains repeated destination filenames.")
    frame.to_csv(output / "mrl_manifest.csv", index=False)
    provenance = {
        "source": "MRL Eye Dataset",
        "source_page": SOURCE_URL,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "images": len(frame),
        "people": int(frame.subject_id.nunique()),
        "label_mapping": "MRL closed=0/open=1 -> project open=0/closed=1",
        "terms": "Consult the original dataset provider; no license is inferred by this importer.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/eyes"))
    args = parser.parse_args()
    print(json.dumps(import_mrl(args.archive, args.output), indent=2))

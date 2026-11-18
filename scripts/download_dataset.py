# /// script
# requires-python = ">=3.12"
# dependencies = ["gdown>=5.2,<6"]
# ///
"""Download NYUv2 RGB/normal pairs and create train, val and test folders.

Run from the repository with: uv run scripts/download_dataset.py
The original test split is preserved. Validation uses 20% of training scenes.
Normal PNGs encode components as 2 * (RGB / 255) - 1; black pixels are invalid.
"""

import argparse
import json
import random
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile, is_zipfile

DATASET_ID = "1Rl-x6GwIXWdRW6LbnSayjIAfqCqygvYQ"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data"
SCENES_PATH = Path(__file__).with_name("nyuv2_scenes.json")


def download_archive(directory: Path) -> Path:
    """Download the authors' prepared archive into a temporary directory."""
    import gdown  # pylint: disable=import-outside-toplevel

    archive = directory / "nyu_dataset.zip"
    result = gdown.download(id=DATASET_ID, output=str(archive), quiet=False)
    if result is None or not is_zipfile(archive):
        raise RuntimeError("Dataset download failed. Run the same command to retry.")
    return archive


def find_pairs(archive: ZipFile) -> dict[str, dict[str, dict[str, str]]]:
    """Find image/normal pairs without relying on the ZIP's top-level folder."""
    pairs: dict[str, dict[str, dict[str, str]]] = {"train": {}, "test": {}}
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts
        if member.is_dir() or len(parts) < 3:
            continue
        split, kind, filename = parts[-3:]
        if split not in pairs or kind not in ("img", "norm"):
            continue
        sample_id = Path(filename).stem
        if not sample_id.isdecimal() or Path(filename).suffix.lower() != ".png":
            continue
        sample = pairs[split].setdefault(sample_id, {})
        if kind in sample:
            raise ValueError(f"Duplicate {kind} for {split}/{sample_id}")
        sample[kind] = member.filename

    for split, samples in pairs.items():
        if not samples:
            raise ValueError(f"No {split} samples found in the archive.")
        for sample_id, sample in samples.items():
            if set(sample) != {"img", "norm"}:
                raise ValueError(f"Missing image/normal pair: {split}/{sample_id}")
    return pairs


def create_splits(
    pairs: dict[str, dict[str, dict[str, str]]],
    scenes: list[str],
    val_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Assign entire training scenes to validation, preserving the test set."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    training_scenes = sorted({scenes[int(key)] for key in pairs["train"]})
    test_scenes = {scenes[int(key)] for key in pairs["test"]}
    if set(training_scenes) & test_scenes:
        raise ValueError(
            "The source training and test sets contain overlapping scenes."
        )
    if len(training_scenes) < 2:
        raise ValueError("At least two training scenes are needed for validation.")
    random.Random(seed).shuffle(training_scenes)
    count = max(
        1, min(len(training_scenes) - 1, round(len(training_scenes) * val_fraction))
    )
    validation_scenes = set(training_scenes[:count])

    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for original_split, samples in pairs.items():
        for sample_id in sorted(samples, key=int):
            scene = scenes[int(sample_id)]
            split = original_split
            if original_split == "train" and scene in validation_scenes:
                split = "val"
            splits[split].append(
                {"id": sample_id, "scene": scene, **samples[sample_id]}
            )
    return splits


def prepare_dataset(
    archive_path: Path,
    root: Path,
    scenes: list[str],
) -> dict[str, int]:
    """Reproduce the current numbered PNG pairs with the fixed scene split."""
    with ZipFile(archive_path) as archive:
        pairs = find_pairs(archive)
        if len(pairs["train"]) != 795 or len(pairs["test"]) != 654:
            raise ValueError("Expected the official NYUv2 split: 795 train / 654 test.")
        ids = [int(key) for samples in pairs.values() for key in samples]
        if len(ids) != len(set(ids)) or set(ids) != set(range(len(scenes))):
            raise ValueError(
                "Archive frame IDs do not match the original scene metadata."
            )
        splits = create_splits(pairs, scenes, val_fraction=0.2, seed=42)
        for split, samples in splits.items():
            filenames = {f"{index}.png" for index in range(len(samples))}
            for kind, directory in (("img", "images"), ("norm", "normals")):
                destination = root / split / directory
                if {path.name for path in destination.glob("*.png")} - filenames:
                    raise ValueError(f"Unexpected PNG files in {destination}.")
                destination.mkdir(parents=True, exist_ok=True)
                for index, sample in enumerate(samples):
                    output = destination / f"{index}.png"
                    content = archive.read(sample[kind])
                    if output.exists():
                        if output.read_bytes() != content:
                            raise ValueError(f"Existing file is different: {output}")
                        continue
                    partial = output.with_suffix(".png.part")
                    partial.write_bytes(content)
                    partial.replace(output)
            print(f"{split}: {len(samples)} pairs -> {root / split}")
    return {split: len(samples) for split, samples in splits.items()}


def main() -> None:
    """Download and organize the dataset using reproducible split settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--archive", type=Path, help="Use an already downloaded nyu_dataset.zip."
    )
    args = parser.parse_args()
    metadata = json.loads(SCENES_PATH.read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="nyuv2-") as temporary:
        archive = args.archive or download_archive(Path(temporary))
        prepare_dataset(archive, args.root, metadata["scenes"])


if __name__ == "__main__":
    main()

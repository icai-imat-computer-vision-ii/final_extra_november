"""Check reproducible extraction without metadata or download-cache folders."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.download_dataset import prepare_dataset


def test_prepare_dataset(tmp_path: Path) -> None:
    """Preserve source bytes, scene separation and numbering across reruns."""

    archive_path = tmp_path / "nyu_dataset.zip"
    scenes = [f"train_room_{i % 249}" for i in range(795)]
    scenes += [f"test_room_{i % 215}" for i in range(654)]
    with ZipFile(archive_path, "w") as archive:
        for index in range(1449):
            split = "train" if index < 795 else "test"
            for kind in ("img", "norm"):
                archive.writestr(
                    f"nyu/{split}/{kind}/{index:06d}.png", f"{kind}:{index}".encode()
                )
        archive.writestr("../../outside.txt", b"ignore unrelated members")

    root = tmp_path / "data"
    counts = prepare_dataset(archive_path, root, scenes)
    assert {path.name for path in root.iterdir()} == {"train", "val", "test"}
    assert counts["test"] == 654
    assert counts["train"] + counts["val"] == 795
    scene_sets: dict[str, set[str]] = {}
    original_ids = set()
    for split, count in counts.items():
        expected_names = {f"{index}.png" for index in range(count)}
        for directory in ("images", "normals"):
            assert {
                p.name for p in (root / split / directory).iterdir()
            } == expected_names
        scene_sets[split] = set()
        for index in range(count):
            image = (root / split / "images" / f"{index}.png").read_bytes()
            source_id = int(image.decode().split(":")[1])
            assert source_id not in original_ids
            original_ids.add(source_id)
            scene_sets[split].add(scenes[source_id])
            assert (root / split / "normals" / f"{index}.png").read_bytes() == (
                f"norm:{source_id}".encode()
            )
    assert original_ids == set(range(1449))
    assert all(
        scene_sets[a].isdisjoint(scene_sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    assert not (tmp_path / "outside.txt").exists()
    assert prepare_dataset(archive_path, root, scenes) == counts

    modified = root / "val" / "images" / "0.png"
    modified.write_bytes(b"user modification")
    with pytest.raises(ValueError, match="different"):
        prepare_dataset(archive_path, root, scenes)
    assert modified.read_bytes() == b"user modification"

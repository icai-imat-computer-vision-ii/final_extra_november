# Surface normal dataset

Run from the project root:

```bash
uv run scripts/download_dataset.py
```

The script downloads the authors' prepared
[NYUv2 RGB/normal archive](https://github.com/baegwangbin/surface_normal_uncertainty)
into a temporary directory and removes the ZIP when finished. It produces only:

```text
data/
├── train/   # 624 pairs: 0.png through 623.png
│   ├── images/
│   └── normals/
├── val/     # 171 pairs: 0.png through 170.png
│   ├── images/
│   └── normals/
└── test/    # 654 pairs: 0.png through 653.png
    ├── images/
    └── normals/
```

The scene split is fixed: seed 42, with 20% of training scenes assigned to
validation. The original test set is preserved. Files are copied byte for byte
from the archive, in numeric source-ID order within each split. No resizing or
normalization is applied to the saved PNGs. Identical existing files are kept;
conflicting files are rejected rather than overwritten.

No `splits.json` or `.downloads/` is created. Each run downloads approximately
1 GB unless you supply a local archive:

```bash
uv run scripts/download_dataset.py --archive /path/to/nyu_dataset.zip
uv run scripts/download_dataset.py --root /path/to/data
```

The PNGs are RGBA, 640 by 480 pixels. Load them with `.convert("RGB")`. Normal
components decode as `2 * (RGB / 255) - 1`; black RGB pixels are invalid and must
be excluded from loss and metrics. Resizing and decoding belong in `dataset.py`.

`nyuv2_scenes.json` supplies the original scene labels needed to reproduce the
split. Its pinned source URL is recorded in the file. Dataset attribution:
[NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html).

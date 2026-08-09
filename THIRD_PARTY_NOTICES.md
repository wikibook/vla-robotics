# Third-Party Notices

This repository is licensed under Apache-2.0 unless a file or directory says
otherwise. The items below are included as source, vendored copies, generated
examples, or runtime/build dependencies and may carry their own notice
requirements.

## Hugging Face LeLab

- Location: `lelab/`
- Source: https://github.com/huggingface/leLab
- Imported commit: `840a970bd48e91ff5787f8748f6d0a5042e00dc0`
- Local source note: `lelab/VENDOR_SOURCE.txt`
- License: Apache-2.0
- Local license copy: `lelab/LICENSE`
- **Modified: yes** — see the "Modifications" section of
  `lelab/VENDOR_SOURCE.txt`

This vendor copy is **not** identical to the upstream commit. It carries a light
frontend theme, SmolVLA fine-tuning from a pretrained base, pinned dependencies,
and has upstream CI and contributor tooling removed. Apache-2.0 section 4(b)
requires that modified files state they were changed; `lelab/VENDOR_SOURCE.txt`
carries that statement and the per-area summary.

The upstream git metadata was removed when vendored. Keep `lelab/LICENSE` and
`lelab/VENDOR_SOURCE.txt` when redistributing this repository or derived source
archives.

## Hugging Face LeRobot

- Source: https://github.com/huggingface/lerobot
- Python dependency: `lerobot`
- Pinned git dependency in `lelab/pyproject.toml`:
  `82dffde7fad11cba91f7916b050fbe7d7eea35ab`
- License: Apache-2.0, with additional bundled MIT notices in the upstream
  LeRobot `LICENSE` file

LeRobot is referenced as a Python dependency in the examples and as a pinned git
dependency by LeLab. Binary or packaged distributions that include LeRobot code
should include the upstream LeRobot license and notices.

## SO-ARM100 / SO-ARM101 Robot Assets

- Locations:
  - `ch06/01-lerobot-record/assets/so101/`
  - `lelab/frontend/public/so-101-urdf/`
- Original source: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
- Original repository license: Apache-2.0
- Local asset notice files:
  - `ch06/01-lerobot-record/assets/so101/NOTICE`
  - `lelab/frontend/public/so-101-urdf/NOTICE`

These assets include URDF/MuJoCo XML and STL mesh files for SO-ARM101. Some XML
files state that they were generated with `onshape-to-robot` from the Onshape
document recorded in the file comments.

## SO-ARM ROS2 URDF Package Wrapper

- Location: `lelab/frontend/public/so-101-urdf/`
- Source: https://github.com/MuammerBay/SO-ARM_ROS2_URDF
- Package metadata license: `BSD` as declared in `package.xml`
- Original asset source referenced by that package:
  https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

The source package metadata uses the non-specific `BSD` label and no full BSD
license text was present in that source repository at the time this notice was
added. The local `package.xml` keeps that upstream metadata instead of silently
rewriting it.

## Frontend npm Dependencies

- Location: `lelab/frontend/package-lock.json`
- License mix observed from the lock file: MIT, Apache-2.0, ISC, BSD-2-Clause,
  BSD-3-Clause, MPL-2.0, BlueOak-1.0.0, Python-2.0, CC-BY-4.0, 0BSD, Zlib, and
  selected dual-license packages.

Notable items for distribution review:

- `jszip` is declared as `(MIT OR GPL-3.0-or-later)`; this project uses the MIT
  option.
- `lightningcss` packages are MPL-2.0.
- `caniuse-lite` is CC-BY-4.0.
- `webgl-constants` did not declare a license in the local lock file.

Generated frontend bundles should be accompanied by dependency notices produced
from the exact lock file used for the build.

## Python Dependencies

Each example is an independent `uv` project. Dependency information is recorded
in each example's `pyproject.toml` and `uv.lock`. Redistributed environments or
containers should include notices for the resolved Python packages from the
corresponding lock file.

# Third-party software and assets

Local code is MIT licensed under David1726. Dependencies and rendered environment
assets retain their own licenses. Upstream source trees and installed dependencies
are downloaded separately and are excluded from source exports.

| Dependency | Use | License / source |
|---|---|---|
| robosuite 1.4 | Panda robot, controllers, nut-placement/stacking environments and scene assets | [MIT](https://github.com/ARISE-Initiative/robosuite) |
| MuJoCo 3.3 | Physics and rendering | [Apache-2.0](https://github.com/google-deepmind/mujoco) |
| LIBERO | Optional pick/place tasks, scene assets and initial states | [MIT](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| RoboRSI | Optional pinned campaign CLI/evidence adapter | [Apache-2.0](https://github.com/nssmd/RoboRSI) |

LIBERO source revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
RoboRSI adapter revision: `e4e9024613e1060614964795206b64f15c422b06`.

The insertion fixtures are defined locally. Reference videos render robosuite
robot and scene assets; they do not transfer ownership of those assets. The
local nut predicate adds release checks around the upstream placement predicate.
Windows compatibility edits are documented in `scripts/prepare_windows_preview.py`.

NumPy, SciPy, Numba, Matplotlib, Pillow, imageio, imageio-ffmpeg, FFmpeg, OpenCV
and the remaining optional packages provide supporting functionality under their
respective licenses. The optional LIBERO stack also uses Gym, BDDL and PyTorch.
No model weights are distributed.

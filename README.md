# LOOKOUT-VO — team cyberkompaspl (MaCVi@WACV 2027 Maritime Visual Odometry)

Monocular VO for boats. Heading from the rotation of the far scene (KLT + inverse-depth windowed BA, essential-matrix chunks on
flir-VIS, VGGT-1B / VGGT-Omega votes); speed from the dense optical flow of the water surface through the law Δv = s·v²
(camera over a water plane, horizon estimated from the water itself), a trust gate and speed rules, and on wide clips a blend with
a temporal CNN trained on the same flow profiles. Final leaderboard entry: **9.15**, file
`submission_lowpow_widemix_conf04_flir_stride5_agree_packed.json`,
SHA-256 `d2dcb411ba3083430af913ab0fd08a1ab99b26645823c611526af00413ac6eb4` (copy in `artifacts/submitted/`).
The uploaded file was produced on **arm64** (Apple M1 Pro, macOS 26.2).

## Quick start

```bash
docker build --platform linux/amd64 -t lookout-vo-repro .
docker run --rm --platform linux/amd64 -v /path/to/lookout-vo-test-v0.1:/data/test:ro \
  -v /path/to/3_challenge:/data/challenge:ro -v "$PWD/out:/out" lookout-vo-repro
```

Needs only the released test package and the organizers' `pack_submission.py`; no GPU, no videos, ≈10 min. The last line is
`IDENTICAL` (byte for byte the submitted file, arm64) or `OK` (every position within 1e-13 m of it, x86-64). The section
*Recomputing the saved outputs from the videos* is only needed to rebuild `artifacts/` itself.

## Layout

| path | what |
|---|---|
| `reproduce.sh` | the whole chain (13 stages) from the saved outputs in `artifacts/` to the final file |
| `Dockerfile` | x86-64 image that runs `reproduce.sh` (numpy only; BLAS kernel and SIMD path fixed) |
| `requirements-arm64.txt` | the same on arm64 macOS |
| `tools/lookout_vo.py` | the tool's water law, trust gate and integration, as the chain re-runs them on the saved raws |
| `tools/lookout_combo/` | stage scripts: heading combination and flir-VIS rule (`course_combo.py`, stage 1), wide slow-down rule (`wide_slowdown.py`, 2), reversing (`wide_reversing.py` with `reversing_rule.py`, `reversing_ramp.py`, 4), U15 / flir-VIS SLR water / native water (`speed_rules.py`, 5–8), low-flow correction (`lowflow_correction.py`, 10), window stride (`flirvis_stride.py`, 12) |
| `flowmodel/` | temporal CNN: seed training (`train_seeds.py`), weight check (`check_weights.py`), wide blend (`wide_mix_build.py`, stage 11) |
| `notebooks/` | GPU steps: VGGT-1B, VGGT-Omega, WaSR-SLR water maps (Colab / Kaggle), temporal-CNN training (Kaggle) |
| `fromvideo/` | per saved input, the code that produced it from the videos (see below) |
| `locks/` | `pip freeze`, Python, GPU, driver, CUDA / cuDNN of the GPU runs |
| `artifacts/` | saved outputs of the tool and the networks, submitted files used as checkpoints, CNN features, predictions and weights; `SHA256SUMS*` |

## Reproducing the final entry from the saved outputs

Inputs besides this repository: the released test package `lookout-vo-test-v0.1` (manifest, `starter/`, `timing/`; the videos are
not read) and the organizers' `pack_submission.py`.

x86-64 (Docker):

```bash
docker build --platform linux/amd64 -t lookout-vo-repro .
docker run --rm --platform linux/amd64 -v /path/to/lookout-vo-test-v0.1:/data/test:ro \
  -v /path/to/3_challenge:/data/challenge:ro -v "$PWD/out:/out" lookout-vo-repro
```

arm64 macOS (Python 3.13):

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-arm64.txt
PY=.venv/bin/python ./reproduce.sh /path/to/lookout-vo-test-v0.1 /path/to/pack_submission.py out
```

The script first checks the SHA-256 of every file in `artifacts/`. Each stage then starts from the previous stage's submitted
file (`artifacts/checkpoints`) and is compared with its own. At the end it prints the SHA-256 of
`out/submission_lowpow_widemix_conf04_flir_stride5_agree_packed.json` and one of:

- `IDENTICAL` — byte for byte the submitted file;
- `OK` — every position within 1e-13 m of the submitted file (another machine's libm / BLAS / SIMD changes only the last bits);
- `FAIL` (exit 1) — anything else.

| stage | what | checkpoint |
|---|---|---|
| 1–4 | heading combination, flir-VIS speed rule, wide slow-down rule Z, reversing with 40-s ramps | `03_zwol_cof40` |
| 5 | U15: clip trust → 1 on smooth, lasting water | `05_u15` |
| 6 | flir-VIS: SLR water where its 20-s window obeys the water law, else 8 Hz | `06_slrcond8` |
| 7 | wide: speed values from the native-resolution water | `07_native` |
| 8–9 | flir-VIS: SLR cond + horizon `slrsel:10`, gate 0.635, stop-ok everywhere; frame-0 sample | `09_frame0` |
| 10–13 | wide low-flow correction, blend with the temporal CNN, flir-VIS 5-s window stride, pack | final file |

Measured:

| machine | stages 1–9 vs checkpoints | final file | time |
|---|---|---|---|
| Apple M1 Pro, macOS 26.2, Python 3.13.7, numpy 2.4.6 (Accelerate) | 2–20 of 20 clips bit for bit, ≤ 1.1e-13 m | `IDENTICAL` | 227 s, 123 MB peak |
| Docker x86-64 under colima / Rosetta on that Mac | 6–20 of 20, ≤ 1.1e-13 m | `OK`, 13/20 bit for bit, ≤ 4.26e-14 m | 8 min |
| Intel Xeon @ 2.00 GHz with AVX-512 (Kaggle), the image's settings | 6–20 of 20, ≤ 1.1e-13 m | `OK`, same file as above (SHA-256 `8a5bd847…`) | 9.5 min |
| Intel Xeon @ 2.20 GHz, AVX2 only (Kaggle), the image's settings | 6–20 of 20, ≤ 1.1e-13 m | `OK`, same file (`8a5bd847…`) | 11.7 min |

The image fixes `OPENBLAS_CORETYPE=Nehalem` and disables numpy's AVX-512 paths: without these two settings the same code on the
same two Xeons gives 2.27e-13 m (AVX-512, OpenBLAS SkylakeX) and 1.99e-13 m (AVX2, OpenBLAS Haswell) — `FAIL`.

Stages 1–6 and 8–9 were first computed on an x86-64 cloud machine (Intel Xeon @ 2.80 GHz, AVX-512, Ubuntu 24.04, Python 3.11.17,
numpy 1.26.4); stage 7 and stages 10–13 on the Mac. Recomputed elsewhere they agree with their checkpoints to ≈1e-13 m but not in
every last bit, so the chain continues from the checkpoints, and the end result does not depend on where stages 1–9 ran.

## Recomputing the saved outputs from the videos

`reproduce.sh` starts from `artifacts/inputs` and `artifacts/flowmodel_predictions`. Each of them was produced as below.
`fromvideo/<input>/` holds the code of the CPU step that produced `artifacts/inputs/<input>`: the files its command imports
(`tools/lookout_vo.py` + `camera/postprocess.py`, `byte_track.py`, `ncnn_infer.py`; for the native water and the E16 / BA runs also
the `tools/lookout_combo/` modules), byte for byte from the development-repository commit that ran, unchanged; `SOURCE.txt` gives
the commit, date, clips and command. The scripts expect the working layout `data/test` (test package) and `data/release` (dev
release) and are run from that directory. Data: the released packages, each checked against its `CHECKSUMS.sha256`. "Bit for bit"
below = recomputed from the video in the listed environment and compared with the saved file.

### CPU: the VO tool (`fromvideo/<input>/tools/lookout_vo.py`)

Environment: Python 3.11, `pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 scipy==1.13.1` (the wheel's FFmpeg,
avcodec 59: another decoder changes the flir-VIS heading). All runs: `--dataset lookout-vo-test-v0.1/release_manifest.json --clips
<clip> --method adaptive --save-raw --output <dir>`; the raw is `<dir>/raw/<clip>.npz`.

| input (`fromvideo/` dir) | clips | extra flags / environment | where it ran | recomputed |
|---|---|---|---|---|
| `test_v4_raw` | 11 flir-VIS | — | x86-64 cloud (above) | not re-run |
| `fvt_hstep_raw` | 11 flir-VIS | — | x86-64 cloud | bit for bit 11/11 (Kaggle Xeon AVX-512, `OPENBLAS_CORETYPE=Prescott`) |
| `test_v5_raw` (+ `test_v5_predictions`) | 8 wide (the 12 other predictions are those of `test_v4`) | first `--full-res --output fullres_test`, then `--speed-from fullres_test --speed-from-modes wide --output test_v5` | x86-64 cloud | wide heading (BA) differs by ≈1e-5 rad on other x86-64 CPUs (Kaggle Skylake), bit for bit only on the cloud CPU; the BA is deterministic on a given machine |
| `dk_slr2_fvt_raw` | 10 flir-VIS (all but 104) | `LOOKOUT_WATER_MASK_DIR=<WaSR-SLR maps>` (below) | Docker `tools/lookout_combo/docker/Dockerfile.x86` | bit for bit 10/10 (same container) |
| `dk_slr2_fvt_raw_clip104` | 104 | as above, an earlier commit | same | bit for bit 1/1 |

### CPU: heading runs for flir-VIS (`lt_half_fx_test`, `lt_ba_fx2_test`; `fromvideo/lt_fx_heading`)

E16 heading and BA on the tool's KLT tracks with the boat-fixed image areas masked: `e16_course.py --save-tracks` →
`fixed_tracks.py` rule `still` → `e16_course.py --fixed-from`, BA cache by `course_combo.ba_fixed`; x86-64 cloud. Not re-run.

### CPU: native-resolution water of the wide clips (`native_full_band_test_raw`, `native_full_test_raw`)

The water profile from every native frame pair at full resolution, summed into the tool's 8-Hz steps; reads the videos and the
8-Hz raws of `test_v5` (every other field of the output is copied from them):

```bash
I=/path/to/this/repo/artifacts/inputs; D=fromvideo/native_full
WIDE=$(python -c "import json; print(','.join(c['key'] for c in json.load(open('data/test/release_manifest.json'))['clips'] if c['camera_mode'] == 'wide'))")
FVT_RAW=$I/test_v5_raw NF_CLIPS=$WIDE NF_BAND=1 NFT_OUT=native_full_band_test_raw PYTHONPATH=$D/tools/lookout_combo:$D/tools:data/release/starter \
  python $D/tools/lookout_combo/native_full.py --native-full
FVT_RAW=$I/test_v5_raw NF_CLIPS=clip_092,clip_097,clip_101,clip_103,clip_105 NFT_OUT=native_full_test_raw PYTHONPATH=$D/tools/lookout_combo:$D/tools:data/release/starter \
  python $D/tools/lookout_combo/native_full.py --native-full
```

The band run covers all 8 wide test clips (stage 7); the full-band run covers 092 097 101 103 105, and stage 10 takes the speed
values from it where its file exists and the 8-Hz values otherwise. Environment: the Mac's `/usr/bin/python3` 3.9.6, numpy 1.26.4,
opencv-python 4.10.0 (avcodec 60): `native_full.py` recomputes all 13 files byte for byte; in the x86-64 container the decoder
differs and the stage-7 routes move by ≤ 3.96 m.

### GPU: the networks (`notebooks/`, environment in `locks/`)

NVIDIA Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 9.27; inference only, frozen public weights.

| artifact | notebook | settings | weights | recomputed |
|---|---|---|---|---|
| `vggt_kf4_w17/` (19 wide + flir-VIS) | `lookout_vggt_colab.ipynb` | windows `kf4_w17` (keyframe every 4 steps, 17 per window) | `facebook/VGGT-1B`, revision `860abec` | bit for bit 2/2 clips (Kaggle) |
| `vggt_omega_kf4_w17/` (8 wide) | `lookout_vggt_omega_colab.ipynb` | `kf4_w17`, fp16, 512 px | `vggt_omega_1b_512.pt`, SHA-256 `c02da418…7934` (needs a Hugging Face token) | all numeric fields bit for bit 1/1 clip (Colab) |
| WaSR-SLR water maps for `dk_slr2_fvt_raw` (11 flir-VIS) | `lookout_wasr_slr_flirvis_colab.ipynb` | run `wasr_v4_slr_flirvis`, class map stored at 960 px | `wasr_slr_v2_rn101.pth`, SHA-256 `1e6cb14d…e74b` | bit for bit 2/2 clips (Kaggle) |

Approximate time on a T4: VGGT-1B ≈3 min per clip, VGGT-Omega ≈5 min per clip, WaSR-SLR 14–16.5 min per flir-VIS clip.

### GPU: the temporal CNN (`artifacts/flowmodel_predictions/`)

Input: per-step flow-profile features of every dev and test clip, `artifacts/flowmodel_features/allcam/clip_*.npz` (`X` per
step; `y` = log GPS speed minus its clip median on dev clips, empty (NaN) on test clips), built by `fromvideo/flowmodel_features/`
from the tool's raws (not re-run). Training: `notebooks/lookout_flowmodel_big_kaggle.ipynb` / `flowmodel/train_seeds.py`
(`FM_DATA=artifacts/flowmodel_features FM_SEED=0..3`) on the 68 training clips (5 folds by clip, checkpoints at epochs 10–60,
variant `allcam_nocam`), ≈46 min per seed on a Kaggle GPU; outputs `big_pred_allcam_nocam{,_s1,_s2,_s3}.npz` + `big_meta_*.json`.
GPU training with mixed precision is not bit-deterministic, so a retrained set differs from the saved predictions.

Weights: the training run kept seed 0's epoch-60 fold models (`artifacts/flowmodel_weights_seed0/`, 6.57 M parameters each); the
other 115 checkpoints of the blend were not saved. `python flowmodel/check_weights.py` runs these 5 models on the test features
(fp32, CPU) and compares them with the saved predictions of that run (`<clip>|60`): largest difference 5.4e-4 over the 20 test
clips (log speed; the predictions' spread is 0.03–0.65), i.e. the fp16 rounding of the GPU run.

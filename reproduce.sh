#!/bin/bash
# Reproduce the final LOOKOUT-VO entry (leaderboard 9.15):
#   submission_lowpow_widemix_conf04_flir_stride5_agree_packed.json, SHA-256 d2dcb411ba3083430af913ab0fd08a1ab99b26645823c611526af00413ac6eb4
# Every stage of the chain is computed, from the saved outputs of the tool and of the models (artifacts/inputs: tool raws, heading
# runs, VGGT / VGGT-Omega windows, native-resolution water; artifacts/flowmodel_predictions: temporal-CNN predictions). The files
# submitted along the way are kept as checkpoints (artifacts/checkpoints): each stage starts from the previous stage's checkpoint, and
# its output is compared with its own checkpoint (clips bit for bit, largest position difference). Stages 1-8 were first computed on
# another machine; here they agree with the checkpoints to ~1e-13 m but not in every last bit (floating-point differences between
# machines), so the chain continues from the checkpoints and the final file is byte-identical. Stages 10-13 were computed on
# macOS arm64 and reproduce bit for bit there (Python 3.13.7, numpy 2.4.6).
#   1-4 course combination + flir-VIS speed rule with stop-ok where the horizon rescue fired (course_combo.test_predictions),
#       wide slow-down rule Z (wide_slowdown.py), wide reversing rule with 40-s ramps (wide_reversing.py)
#   5   U15: clip trust -> 1 on smooth, lasting water (speed_rules.py --u-test / --test-variant)
#   6   flir-VIS: SLR water where its 20-s window obeys the water law, else 8 Hz (--fv-test-var)
#   7   wide: speed values from the native-resolution water (--native-test)
#   8   flir-VIS: SLR cond + horizon slrsel:10, gate threshold 0.635, stop-ok everywhere (--fv-test-var)
#   9   frame-0 sample: the first step extrapolated back to frame 0
#   10  wide low-flow correction (lowflow_correction.py, LOWPOW=3.25:0.3:2)
#   11  wide blend with the temporal CNN, per-step weight 0.4 s0^2/(s0^2+spread^2), s0 = 0.0247 (wide_mix_build.py)
#   12  flir-VIS slow-down rule with 20-s windows every 5 s where the 5-s and 10-s grids agree (flirvis_stride.py)
#   13  pack with the organizers' pack_submission.py, SHA-256 check
# Usage: ./reproduce.sh TEST_RELEASE_DIR PACK_SUBMISSION_PY [OUT_DIR]
#   TEST_RELEASE_DIR   lookout-vo-test-v0.1 (release_manifest.json, starter/, timing/; the videos are not read)
#   PACK_SUBMISSION_PY the organizers' 3_challenge/pack_submission.py
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd); TEST=$(cd "$1" && pwd); PACK=$2; OUT=${3:-$REPO/out}
A=$REPO/artifacts; I=$A/inputs; C=$A/checkpoints; PY=${PY:-python3}; W=$(mktemp -d); mkdir -p "$OUT"
trap 'rm -rf "$W"' EXIT
sha256() { if command -v sha256sum > /dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
( cd "$A" && sha256 -c --quiet SHA256SUMS SHA256SUMS_inputs ) || { echo "artifacts: SHA-256 check FAILED"; exit 1; }
echo "artifacts: SHA-256 OK"
# the working layout the scripts expect (relative paths data/, runs/, tools/)
mkdir -p "$W/data" "$W/runs/test_v4" "$W/runs/test_v5" "$W/runs/fvt_hstep" "$W/runs/dk_slr2_fvt" "$W/runs/native/fullt" "$W/runs/vggt" \
  "$W/runs/vggt_omega" "$W/s"
ln -s "$TEST" "$W/data/test"; ln -s "$REPO/tools" "$W/tools"
ln -s "$I/test_v4_raw" "$W/runs/test_v4/raw"; ln -s "$I/test_v5_raw" "$W/runs/test_v5/raw"; ln -s "$I/test_v5_predictions" "$W/runs/test_v5/predictions"
ln -s "$I/fvt_hstep_raw" "$W/runs/fvt_hstep/raw"; ln -s "$I/dk_slr2_fvt_raw" "$W/runs/dk_slr2_fvt/raw"; ln -s "$I/native_full_test_raw" "$W/runs/native/fullt/raw"
ln -s "$I/vggt_kf4_w17" "$W/runs/vggt/kf4_w17"; ln -s "$I/vggt_omega_kf4_w17" "$W/runs/vggt_omega/kf4_w17"
for d in lt_half_fx_test lt_ba_fx2_test; do ln -s "$I/$d" "$W/runs/$d"; done
cd "$W"; S=$W/s; export PYTHONPATH=tools/lookout_combo:tools:data/test/starter
pack() { rm -f "$2"; PYTHONPATH=data/test/starter "$PY" "$PACK" --dataset data/test/release_manifest.json --predictions "$1" --output "$2" > /dev/null; }
split() { rm -rf "$2"; mkdir -p "$2/predictions"; "$PY" -c "import json,sys; S=json.load(open(sys.argv[1])); [json.dump(v,open(f'{sys.argv[2]}/predictions/{k}.json','w')) for k,v in S.items()]" "$1" "$2"; }
check() {  # check STAGE OUT.json CHECKPOINT.json [CLIP,...]: clips bit for bit and the largest difference vs the checkpoint (CLIP: left out)
  "$PY" - "$@" <<'PY'
import json, sys, numpy as np
st, a, b = sys.argv[1:4]; skip = set(sys.argv[4].split(',')) if len(sys.argv) > 4 else set()
A, B = json.load(open(a)), json.load(open(b)); ks = sorted(k for k in B if k not in skip); same, worst = 0, 0.0
for k in ks:
    assert A[k]['times'] == B[k]['times'] and A[k]['frame_indices'] == B[k]['frame_indices'], k
    pa, pb = np.asarray(A[k]['positions'], float), np.asarray(B[k]['positions'], float); same += bool(np.array_equal(pa, pb)); worst = max(worst, float(np.abs(pa - pb).max()))
print(f"stage {st}: vs checkpoint {b.split('/')[-1]}: {same}/{len(ks)} clips bit for bit, largest difference {worst:.2g} m" + (f" ({', '.join(sorted(skip))} left out)" if skip else ''))
PY
}
echo "1-4 course + flir-VIS rule, Z, reversing ramp (from the inputs)"
"$PY" -c "
import sys; sys.argv=['x']
import course_combo as CC
CC.test_predictions(out='$S/run1',mE=0.60,mB=0.45,mV=0.55*1.8,wideV=0.80,wideO=(1.0,0.5),fscale_flirvis=1.8,c5_flirvis=0.0,offset_flirvis='runs/fvt_hstep/raw',stop_min_good=0.5,stop_hz='best',stop_ok='rescued')" > "$OUT/log_01.txt" 2>&1
pack "$S/run1" "$S/01.json"
SUB="$S/01.json" OUT="$S/run2" "$PY" tools/lookout_combo/wide_slowdown.py > "$OUT/log_02.txt" 2>&1; pack "$S/run2" "$S/02.json"
L=40 SUB="$S/02.json" OUT="$S/run4" "$PY" tools/lookout_combo/wide_reversing.py > "$OUT/log_04.txt" 2>&1; pack "$S/run4" "$S/04.json"
check 1-4 "$S/04.json" "$C/03_zwol_cof40.json" clip_109   # the checkpoint predates the stop-ok rule, which changes only 109 (checked at 5)
echo "5 U15"
TVC=$(U_RULE='step:15:0.06:0:1:0.5' "$PY" tools/lookout_combo/speed_rules.py --u-test 2>/dev/null | sed -n 's/^TV_ALPHA_CLIPS=//p' | tail -1)
SUB="$S/04.json" TV_ALPHA_CLIPS="$TVC" TV_OUT="$S/05_raw.json" "$PY" tools/lookout_combo/speed_rules.py --test-variant > "$OUT/log_05.txt" 2>&1
split "$S/05_raw.json" "$S/run5"; pack "$S/run5" "$S/05.json"; check 5 "$S/05.json" "$C/05_u15.json"
echo "6 flir-VIS SLR water (from checkpoint 5)"
SUB="$C/05_u15.json" FVTV='c=runs/dk_slr2_fvt/raw+runs/fvt_hstep/raw@-@0.35@cond' FVTV_WRITE=c FVTV_OUT="$S/06.json" \
  "$PY" tools/lookout_combo/speed_rules.py --fv-test-var > "$OUT/log_06.txt" 2>&1
check 6 "$S/06.json" "$C/06_slrcond8.json"
echo "7 wide native-resolution water (from checkpoint 6)"
SUB="$C/06_slrcond8.json" NT_SPEED_ONLY="$I/native_full_band_test_raw" NT_OUT="$S/07.json" \
  "$PY" tools/lookout_combo/speed_rules.py --native-test > "$OUT/log_07.txt" 2>&1
check 7 "$S/07.json" "$C/07_native.json"
echo "8 flir-VIS SLR cond + slrsel:10, threshold 0.635, stop-ok everywhere (on the stage-5 flir-VIS rule; wide from checkpoint 7)"
LOOKOUT_STOP_THR=0.635 FV_STOP_OK=all SUB="$C/05_u15.json" FVTV='sel=runs/dk_slr2_fvt/raw+runs/fvt_hstep/raw@-@0.35@cond@slrsel:10' \
  FVTV_WRITE=sel FVTV_OUT="$S/08_fv.json" "$PY" tools/lookout_combo/speed_rules.py --fv-test-var > "$OUT/log_08.txt" 2>&1
echo "9 frame-0 sample"
"$PY" - "$S" "$C/07_native.json" <<'PY'
import json, sys, numpy as np
S = sys.argv[1]; W = json.load(open(sys.argv[2])); F = json.load(open(f'{S}/08_fv.json'))
man = {c['key']: c for c in json.load(open('data/test/release_manifest.json'))['clips']}
out = {}
for k in W:
    v = F[k] if man[k]['camera_mode'] == 'flir-VIS' else W[k]
    t0 = json.load(open(f'data/test/timing/{k}.json'))['times'][0]; T = np.asarray(v['times'], float); P = np.asarray(v['positions'], float)
    f0 = P[0] - (P[1] - P[0]) / (T[1] - T[0]) * (T[0] - t0)
    out[k] = dict(v, times=[t0] + v['times'], positions=[f0.tolist()] + v['positions'], frame_indices=[0] + v['frame_indices'])
json.dump(out, open(f'{S}/09.json', 'w'), separators=(',', ':'), allow_nan=False)
PY
check 8-9 "$S/09.json" "$C/09_frame0.json"
echo "10 wide low-flow correction (from checkpoint 9)"
LOWPOW=3.25:0.3:2 BASE="$C/09_frame0.json" OUT="$S/10.json" PYTHONPATH="$REPO/tools:$REPO/tools/lookout_combo" "$PY" "$REPO/tools/lookout_combo/lowflow_correction.py" > "$OUT/log_10.txt" 2>&1
echo "11 wide blend with the temporal CNN"
P=$A/flowmodel_predictions
WM_BASE="$S/10.json" WM_CONF_S0=0.0247 "$PY" "$REPO/flowmodel/wide_mix_build.py" \
  "$P/big_pred_allcam_nocam.npz,$P/big_pred_allcam_nocam_s1.npz,$P/big_pred_allcam_nocam_s2.npz,$P/big_pred_allcam_nocam_s3.npz" \
  0.4 "$S/11_raw.json" "$S/run11" > "$OUT/log_11.txt" 2>&1
pack "$S/run11/predictions" "$S/11.json"
echo "12 flir-VIS slow-down rule, 5-s window stride with agreement gate"
BASE930="$C/09_frame0.json" LOOKOUT_STOP_THR=0.635 FV_STOP_OK=all AGREE=0.05 \
  "$PY" "$REPO/tools/lookout_combo/flirvis_stride.py" "$REPO" 5 "$S/11.json" "$S/12.json" > "$OUT/log_12.txt" 2>&1
echo "13 pack"
split "$S/12.json" "$S/run13"
F="$OUT/submission_lowpow_widemix_conf04_flir_stride5_agree_packed.json"; rm -f "$F"
PYTHONPATH=data/test/starter "$PY" "$PACK" --dataset data/test/release_manifest.json --predictions "$S/run13/predictions" --output "$F"
H=$(sha256 "$F" | cut -d' ' -f1); echo "SHA-256 $H"
[ "$H" = d2dcb411ba3083430af913ab0fd08a1ab99b26645823c611526af00413ac6eb4 ] && { echo "IDENTICAL to the submitted final entry"; exit 0; }
# another machine's libm / BLAS / SIMD: the same numbers up to the last bits -- pass within 1e-13 m of the submitted positions
"$PY" - "$F" "$A/submitted/submission_lowpow_widemix_conf04_flir_stride5_agree_packed.json" <<'PY'
import json, sys, numpy as np
A, B = (json.load(open(f)) for f in sys.argv[1:3]); assert sorted(A) == sorted(B), 'clip lists differ'
same, worst = 0, 0.0
for k in B:
    assert A[k]['times'] == B[k]['times'] and A[k]['frame_indices'] == B[k]['frame_indices'], k
    d = float(np.abs(np.asarray(A[k]['positions'], float) - np.asarray(B[k]['positions'], float)).max()); same += d == 0; worst = max(worst, d)
ok = worst <= 1e-13
print(f"{'OK' if ok else 'FAIL'}: vs the submitted final entry {same}/{len(B)} clips bit for bit, largest position difference {worst:.3g} m (limit 1e-13 m)")
sys.exit(0 if ok else 1)
PY

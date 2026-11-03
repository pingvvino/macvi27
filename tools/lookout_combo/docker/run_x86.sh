#!/bin/sh
# Run a python script of this repo in the x86-64 container (Dockerfile.x86), e.g. the VO tool with the SLR water masks:
#   docker build --platform linux/amd64 -f tools/lookout_combo/docker/Dockerfile.x86 -t lookout-x86:py311-cv41084 tools/lookout_combo/docker
#   LOOKOUT_MOUNTS="/data /masks" LOOKOUT_ENV=LOOKOUT_WATER_MASK_DIR=/masks/wasr_v4_slr_flirvis \
#     tools/lookout_combo/docker/run_x86.sh fromvideo/b2976ba/tools/lookout_vo.py --dataset /data/lookout-vo-test-v0.1/release_manifest.json ...
# LOOKOUT_MOUNTS: host directories mounted at the same path (give absolute paths); the current directory is always mounted.
WT="$(pwd)"; M=""
for d in $LOOKOUT_MOUNTS; do M="$M -v $d:$d"; done
exec docker run --rm --platform linux/amd64 -v "$WT:$WT" $M -w "$WT" -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONDONTWRITEBYTECODE=1 \
  ${LOOKOUT_ENV:+-e "$LOOKOUT_ENV"} lookout-x86:py311-cv41084 \
  python -c "import cv2, sys, runpy; cv2.setNumThreads(2); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" "$@"

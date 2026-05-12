#!/usr/bin/env bash
# Exp 3 — basis-FWHM gap rebuild.
#
# Vrabel smoke test showed basis_fwhm_mismatch ~0.032 nm vs ~0.008 nm for
# BHVO-2. Hypothesis: the 8-FWHM canonical grid (0.05, 0.10, 0.17, 0.25,
# 0.50, 0.71, 1.00, 1.67) is too sparse in the 0.05-0.10 range where
# Vrabel echelle ICCD's RP-class falls. Add 4 FWHMs: 0.03, 0.04, 0.06,
# 0.08.
#
# Build runs on vasp-02 (CPU-only, ~30 min). Output goes directly to NFS.
# Re-running the canonical builder is a no-op for existing files.

set -euo pipefail
echo "=== Exp 3 basis-FWHM rebuild @ $(date '+%Y-%m-%d %H:%M:%S') ==="

# Sync repo + verify build script + db present
ssh root@10.0.0.21 'test -d /scratch/cf-libs-build-temp || mkdir -p /scratch/cf-libs-build-temp'
rsync -aL --exclude=.git --exclude=output --exclude=.venv --exclude=data --exclude=cache_basis \
  /home/brian/code/CF-LIBS-improved/ root@10.0.0.21:/scratch/cf-libs-build-temp/
ssh root@10.0.0.21 'cd /scratch/cf-libs-build-temp && ln -sfn /scratch/cf-libs-bench/repo/.venv .venv'

# Trigger
ssh root@10.0.0.21 'cd /scratch/cf-libs-build-temp && \
  setsid nohup bash -c "export PYTHONPATH=. JAX_PLATFORMS=cpu; \
    .venv/bin/python scripts/build_basis_library.py --fwhm 0.03 0.04 0.06 0.08 2>&1 \
    | tee /tmp/exp003-basis-rebuild.log" </dev/null >/tmp/exp003-wrap.log 2>&1 & disown; sleep 2; \
  pgrep -af build_basis_library | head -2'

echo "Launched. Monitor: ssh root@10.0.0.21 tail -f /tmp/exp003-basis-rebuild.log"
echo "Outputs at /cluster/shared/cf-libs-bench/basis_libraries/basis_fwhm_{0.03,0.04,0.06,0.08}nm.h5"

#!/usr/bin/env bash
#
# Cross-node EFA SRD send-completion locality probe.
#
# Builds the monarch_rdma library test binary once, then launches it as two
# SLURM tasks (one per node). Rank 0 (SLURM_PROCID 0) acts as the responder,
# rank 1 as the initiator; they exchange QP endpoint info + the destination
# buffer through a shared-filesystem rendezvous directory. The initiator writes
# across the real EFA fabric -- a control WRITE + RDMA readback (proving the
# data landed on the *other* host), a bad-rkey WRITE, and an out-of-bounds
# WRITE -- and prints the verdict. A bad-rkey WRITE that comes back
# IBV_WC_REM_ACCESS_ERR proves the send completion waited for the remote NIC,
# i.e. SRD send completions are NOT local-only.
#
# Usage:
#   PART=h100 ./run_efa_srd_xnode_probe.sh        # auto-allocates 2 nodes via salloc
#   # ...or from inside an existing >=2-node allocation:
#   salloc -N2 --gpus-per-node=1 -p h100
#   ./run_efa_srd_xnode_probe.sh
#
# Env overrides:
#   PART   SLURM partition to allocate on when not already inside a job (default: h100)
#   QOS    SLURM quality-of-service for the salloc (default: h100_dev, for faster placement)
#   REPO   monarch repo root (default: /storage/home/$USER/monarch)
#   RDVZ   rendezvous dir on a shared filesystem (default: $HOME/efa_probe_rdvz)
#
# Host memory only -- no CUDA needed; the GPU request just lands the job on
# EFA-capable nodes.

set -uo pipefail

PART="${PART:-h100}"
QOS="${QOS:-h100_dev}"
REPO="${REPO:-/storage/home/$USER/monarch}"
TEST="backend::ibverbs::efa_queue_pair::tests::efa_srd_send_completion_locality_xnode"

command -v cargo >/dev/null 2>&1 || {
    echo "ERROR: 'cargo' not found in PATH. Load your Rust toolchain (e.g. source ~/.cargo/env" >&2
    echo "       or 'module load rust') and re-run, or run this on a node that has cargo." >&2
    exit 1
}
command -v srun >/dev/null 2>&1 || {
    echo "ERROR: 'srun' not found; this script must run on a SLURM cluster." >&2
    exit 1
}

cd "$REPO" || {
    echo "ERROR: monarch repo not found at '$REPO' (override with REPO=/path/to/monarch)." >&2
    exit 1
}

# --- 1. Build the lib test binary once (shared FS; do NOT build on both nodes
#        concurrently, which would race on target/). --lib scopes to the crate's
#        unit tests, skipping the feature-gated integration tests in tests/. -----
echo ">> [1/3] Building the monarch_rdma lib test binary ..."
build_out=$(cargo test -p monarch_rdma --lib "$TEST" --no-run 2>&1) || {
    echo "$build_out" >&2
    echo "ERROR: build failed (see output above)." >&2
    exit 1
}
echo "$build_out" | grep -E "Finished|Executable" || true

# Prefer the exact path cargo printed; fall back to the newest matching binary.
BIN=$(printf '%s\n' "$build_out" | sed -nE 's/.*Executable[^(]*\((.*)\)/\1/p' | tail -1)
if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
    BIN=$(ls -t target/debug/deps/monarch_rdma-* 2>/dev/null | grep -v '\.' | head -1 || true)
fi
if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
    echo "ERROR: could not locate the built test binary under target/debug/deps/." >&2
    exit 1
fi
echo ">> Test binary: $BIN"

# --- 2. Fresh rendezvous dir on the shared filesystem (both nodes must see it). -
RDVZ="${RDVZ:-$HOME/efa_probe_rdvz}"
rm -rf "$RDVZ"
mkdir -p "$RDVZ"
echo ">> [2/3] Rendezvous dir (wiped fresh): $RDVZ"

# --- 3. Run one task per node: rank 0 = responder, rank 1 = initiator. ----------
# The `env ...` wrapper sets EFA_PROBE_RENDEZVOUS for each task regardless of
# srun's export settings. -N2/--ntasks=2 pins exactly two nodes/tasks even if the
# surrounding allocation is larger (so we never spawn extra initiators).
run=(srun -N2 --ntasks=2 --ntasks-per-node=1
     env EFA_PROBE_RENDEZVOUS="$RDVZ"
     "$BIN" --exact "$TEST" --nocapture --test-threads=1)

echo ">> [3/3] Launching the cross-node probe ..."
rc=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo ">> using the current SLURM allocation (job $SLURM_JOB_ID)"
    "${run[@]}" || rc=$?
else
    echo ">> no allocation detected; requesting 2 nodes on partition '$PART' (qos '$QOS') via salloc"
    salloc -N2 --gpus-per-node=1 -p "$PART" --qos "$QOS" "${run[@]}" || rc=$?
fi

echo
if (( rc == 0 )); then
    echo ">> DONE (exit 0). See the 'VERDICT (cross-node)' block above:"
    echo ">>   'NOT local-only' => a bad-rkey WRITE to the remote host surfaced"
    echo ">>   IBV_WC_REM_ACCESS_ERR on the initiator, so the send completion waited"
    echo ">>   for the remote NIC. (A 'SKIP ...' line instead means the node had no"
    echo ">>   EFA verbs stack -- try a different partition/node.)"
else
    echo ">> FAILED (exit $rc). Check the output above: an assertion (would mean the"
    echo ">> completion looked local-only), a rendezvous timeout, or a SKIP."
fi
exit $rc

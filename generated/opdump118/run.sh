#!/bin/bash
# Exhaustive BF16 TT-NN output dumps over one build (measurement only, not a PR commit).
#   run.sh parents   every op at this tree (the PRs' branch point)
#   run.sh heads     each PR's head: its JIT-compiled files (generated/opdump118/<pr>/) over this tree, one at a time
# Each dump logs "DUMP <op> <tag> <base64 zlib words>"; a fresh JIT cache per dump, so no kernel is reused.
set -u
cd "$(dirname "$0")/../.."
here=generated/opdump118
test=tests/ttnn/unit_tests/operations/eltwise/test_bf16_output_dump.py
dirs="tt_metal/hw/ckernels tt_metal/hw/inc tt_metal/tt-llk/common $(ls -d tt_metal/tt-llk/tt_llk_*)"
DUMP_SPEC_B64="$(base64 -w0 "$here/spec.json")"
export DUMP_SPEC_B64

dump() {  # ops tag
    export TT_METAL_CACHE="/tmp/ttcache-$2"
    rm -rf "$TT_METAL_CACHE" "$HOME/.cache/tt-metal-cache"
    DUMP_OPS="$1" DUMP_TAG="$2" pytest -rA -p no:randomly --timeout 1800 "$test" < /dev/null
    echo "DUMPEXIT $2 $?"
}

case "${1:-}" in
parents)
    dump "$(cut -f2 "$here/heads.tsv" | paste -sd:)" wh-parent
    ;;
heads)
    tar cf /tmp/opdump-pristine.tar $dirs
    while IFS=$'\t' read -r pr op; do
        rm -rf $dirs && tar xf /tmp/opdump-pristine.tar
        cp -a "$here/$pr/." .
        echo "OVERLAY $pr $(cd "$here/$pr" && find . -type f | wc -l) files"
        dump "$op" "$pr-wh-head"
    done < "$here/heads.tsv"
    rm -rf $dirs && tar xf /tmp/opdump-pristine.tar
    ;;
*)
    echo "usage: run.sh parents|heads" >&2
    exit 2
    ;;
esac

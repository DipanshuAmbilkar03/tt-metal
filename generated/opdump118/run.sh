#!/bin/bash
# Exhaustive BF16 TT-NN output dumps over one build (measurement only, not a PR commit).
#   run.sh parents   every op at this tree (the PRs' branch point)
#   run.sh heads     each PR's head: its JIT-compiled files (generated/opdump118/<pr>/) over this tree, one at a time
# A wheel run compiles kernels from the installed ttnn package's tt_metal/, not TT_METAL_HOME, so a head's files go
# into every root that holds tt_metal/hw/ckernels: this checkout and the installed package.
# Each dump logs "DUMP <op> <tag> <base64 zlib words>"; a fresh JIT cache per dump, so no kernel is reused.
set -u
cd "$(dirname "$0")/../.."
work="$PWD"
here=generated/opdump118
test=tests/ttnn/unit_tests/operations/eltwise/test_bf16_output_dump.py
package="$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("ttnn").origin))')"
roots=()
for root in "$work" "$package" "$package/.."; do
    [[ -d "$root/tt_metal/hw/ckernels" ]] && roots+=("$(cd "$root" && pwd)")
done
mapfile -t roots < <(printf '%s\n' "${roots[@]}" | sort -u)
echo "ROOTS ${roots[*]}"
DUMP_SPEC_B64="$(base64 -w0 "$here/spec.json")"
export DUMP_SPEC_B64

dump() {  # ops tag
    export TT_METAL_CACHE="/tmp/ttcache-$2"
    rm -rf "$TT_METAL_CACHE" "$HOME/.cache/tt-metal-cache"
    DUMP_OPS="$1" DUMP_TAG="$2" pytest -rA -p no:randomly --timeout 1800 "$test" < /dev/null
    echo "DUMPEXIT $2 $?"
}

kernel_dirs() {  # root: the JIT-compiled source dirs under it
    (cd "$1" && ls -d tt_metal/hw/ckernels tt_metal/hw/inc tt_metal/tt-llk/common tt_metal/tt-llk/tt_llk_* 2>/dev/null)
}

case "${1:-}" in
parents)
    dump "$(cut -f2 "$here/heads.tsv" | paste -sd:)" wh-parent
    ;;
heads)
    for index in "${!roots[@]}"; do
        (cd "${roots[$index]}" && tar cf "/tmp/opdump-pristine-$index.tar" $(kernel_dirs .))
    done
    restore() {
        for index in "${!roots[@]}"; do
            (cd "${roots[$index]}" && rm -rf $(kernel_dirs .) && tar xf "/tmp/opdump-pristine-$index.tar")
        done
    }
    while IFS=$'\t' read -r pr op; do
        restore
        landed=0
        for root in "${roots[@]}"; do
            cp -a "$work/$here/$pr/." "$root/"
            first="$(cd "$work/$here/$pr" && find . -type f | sort | head -1)"
            cmp -s "$work/$here/$pr/$first" "$root/$first" && landed=$((landed + 1))
        done
        echo "OVERLAY $pr $(cd "$work/$here/$pr" && find . -type f | wc -l) files in $landed of ${#roots[@]} roots"
        dump "$op" "$pr-wh-head"
    done < "$here/heads.tsv"
    restore
    ;;
*)
    echo "usage: run.sh parents|heads" >&2
    exit 2
    ;;
esac

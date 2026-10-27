// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_atan_bf16_replay_window.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_atan_bf16_horner.h"
#include "ckernel_sfpu_atan_bf16_mirrored_terminals.h"
#include "ckernel_sfpu_atan_bf16_reciprocal_complement_core.h"
namespace ckernel::sfpu::bf16::atan {
// SFPU state this kernel overwrites on Blackhole and does not restore. A later op that reads any of it
// must program it in its own init; one that skips its init, or relies on load-macro or replay state it
// did not program, runs with this kernel's.
// - init_reciprocal_complement: vConstFloatPrgm0 (2.0, sfpu_reciprocal_init) and the SFPLOADMACRO
//   configuration: the instruction templates written by the SFPSETSGN and SFP_STOCH_RND below, macro
//   sequence 0 (kMacroSequenceBits, SFPCONFIG 4) and the load-macro misc settings (SFPCONFIG 8).
// - calculate_reciprocal_complement, on every call: LREG4 (the complement constant), LREG5 (the NaN-row
//   bitmap), LREG6 and LREG7 (c1 and c2), LREG0-3 as scratch, and replay slots 0..kBodySlots+3 (the row
//   loads its two leading coefficients too), which it records again on each call. The DEST row counter is set back to
//   row 0 of the tile.
// Each row loads and stores through ADDR_MOD_7, which every SFPU init holds at zero, and an INCRWC after it
// steps DEST: ADDR_MOD_6 belongs to the FPU ops (matmul) a kernel may run between this init and its call.
template <class Config>
inline void init_reciprocal_complement() {
    sfpu_reciprocal_init<false>();
}
template <class Config, int Iterations = 32>
inline void calculate_reciprocal_complement() {
    static_assert(Iterations == 32);
    static_assert(Config::kDegree == 3);
#pragma GCC unroll 32
    for (int d = 0; d < 32; ++d) {
        sfpi::vFloat original = sfpi::dst_reg[d];
        sfpi::vFloat reduced;
        sfpi::vFloat coordinate = ttpoly::atan::reciprocal_complement_reduce(
            original, reduced, [](sfpi::vFloat x) __attribute__((always_inline)) {
                return sfpu_reciprocal_iter<2>(x);
            });
        sfpi::vFloat result = ttpoly::atan::eval_polynomial<3>(Config::kCoefficients, coordinate);
        result = ttpoly::atan::reciprocal_complement_expand(original, reduced, result);
        ttpoly::atan::nan_union_class_terminal<1>(sfpi::dst_reg[d].template mode<sfpi::DataLayout::U16>(), result);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = result;
    }
}
}  // namespace ckernel::sfpu::bf16::atan

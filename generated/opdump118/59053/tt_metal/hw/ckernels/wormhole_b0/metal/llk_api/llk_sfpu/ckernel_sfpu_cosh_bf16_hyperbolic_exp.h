// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_cosh_bf16_replay_window.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_cosh_bf16_hyperbolic_exp_core.h"
namespace ckernel::sfpu::bf16::cosh {
template <typename Config, int Iterations = 32>
inline void calculate_hyperbolic_exp() {
    static_assert(Iterations == 32 && Config::kDegree == 4);
    static_assert(Config::kBare);
    sfpi::l_reg[sfpi::LRegs::LReg7] = sfpi::vFloat(__builtin_bit_cast(float, Config::kMultiplierBits));
    sfpi::vFloat multiplier = sfpi::l_reg[sfpi::LRegs::LReg7];
    sfpi::l_reg[sfpi::LRegs::LReg6] =
        sfpi::vFloat(__builtin_bit_cast(float, Config::kComposeScaleBits ^ (Config::kOdd ? 0x80000000u : 0u)));
    sfpi::vFloat scale = sfpi::l_reg[sfpi::LRegs::LReg6];
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat x = sfpi::dst_reg[row];
        x = sfpi::setsgn(x, 0);
        sfpi::l_reg[sfpi::LRegs::LReg5] = x;
        sfpi::vFloat saved = sfpi::l_reg[sfpi::LRegs::LReg5];
        sfpi::vFloat y = ttpoly::cosh::hyperbolic_exp_wh<Config>(saved, multiplier, scale, saved);
        y = ttpoly::cosh::hyperbolic_wh_terminal<Config>(saved, y);
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = y;
    }
}
}  // namespace ckernel::sfpu::bf16::cosh

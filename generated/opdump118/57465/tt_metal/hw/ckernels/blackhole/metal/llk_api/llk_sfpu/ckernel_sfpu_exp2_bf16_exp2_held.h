// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp2_bf16_replay_window.h"
#include "sfpu/ckernel_sfpu_exp2_bf16_exp2_held.h"
// The exponent-ALU total form one row at a time, with every constant in L3..L7.
// One row needs three working registers, so the multiplier and the coefficients
// fit in LREGs and the setup sets nothing.
namespace ckernel::sfpu::bf16::exp2 {
template <typename Config>
inline void init_exp2_held() {}
}  // namespace ckernel::sfpu::bf16::exp2

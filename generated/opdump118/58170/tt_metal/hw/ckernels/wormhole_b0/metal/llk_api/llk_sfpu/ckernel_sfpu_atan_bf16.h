// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct AtanBf16Config {
    static constexpr unsigned kDegree = 0x00000003u;
    // Fit: minimax degree-3 polynomial on [0, 1], 1 segment, max pure (continuous) ULP 0.539.
    static constexpr float kCoefficients[] = {
        __builtin_bit_cast(float, 0x3f800000u),
        __builtin_bit_cast(float, 0xbea7be2cu),
        __builtin_bit_cast(float, 0x3e232346u),
        __builtin_bit_cast(float, 0xbd3e731cu)};
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_atan_bf16_reciprocal_complement.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format, as BF16 for
// Float16_b and the block formats sharing its exponent.
inline bool bf16_dest_atan() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    const std::uint32_t format = ckernel::math::src_zero_flag_srcb_fmt;
    return format == static_cast<std::uint32_t>(DataFormat::Float16_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp8_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp4_b) ||
           format == static_cast<std::uint32_t>(DataFormat::Bfp2_b);
#else
    return false;
#endif
}
// Each call records replay slots 0-31, overwriting what an init recorded there (matmul and reduce
// record from replay_buf_offset = 16).
template <int ITERATIONS = 32>
__attribute__((noinline)) inline void calculate_atan_bf16() {
    ckernel::sfpu::bf16::atan::calculate_reciprocal_complement<ckernel::sfpu::AtanBf16Config, ITERATIONS>();
}
inline void init_atan_bf16() { ckernel::sfpu::bf16::atan::init_reciprocal_complement<ckernel::sfpu::AtanBf16Config>(); }

}  // namespace ckernel::sfpu

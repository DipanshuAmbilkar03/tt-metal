// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Selected reciprocal coordinate and reconstruction. Callers retain terminals,
// coefficient ownership, rounding and traversal.
namespace ckernel::sfpu::ttpoly::atan {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <class Reciprocal>
__attribute__((always_inline)) inline vFloat reciprocal_complement_reduce(
    vFloat original, vFloat& reduced, Reciprocal reciprocal) {
    vFloat abs_x = setsgn(original, 0);
    vFloat inverse = reciprocal(abs_x);
    reduced = abs_x;
    v_if(abs_x > 1.0f) { reduced = inverse; }
    v_endif;
    return reduced * reduced;
}
__attribute__((always_inline)) inline vFloat reciprocal_complement_expand(
    vFloat original, vFloat reduced, vFloat result) {
    constexpr float complement = 1.5707963267948966f;
    vFloat core_value = reduced * result;
    vFloat magnitude = core_value;
    v_if(setsgn(original, 0) > 1.0f) { magnitude = complement - core_value; }
    v_endif;
    return copysgn(magnitude, original);
}
}  // namespace ckernel::sfpu::ttpoly::atan

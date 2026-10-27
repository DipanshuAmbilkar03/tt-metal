// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Selected P4 exp/reciprocal bodies. Callers own init, traversal and stores.
namespace ckernel::sfpu::ttpoly::cosh {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <typename Config>
__attribute__((always_inline)) inline vFloat hyperbolic_exp_wh(
    vFloat x, vFloat multiplier, vFloat compose_scale, [[maybe_unused]] vFloat original) {
    l_reg[LRegs::LReg4] = vFloat(126.0f);
    vFloat bias_register = l_reg[LRegs::LReg4];
    vFloat xlog2 = x * multiplier + bias_register;
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);
    vInt ep = exexp(z, ExponentMode::Biased);
    vMag fm = exman(z);
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest);
    vFloat c4 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[4]);
    vFloat c3 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[3]);
    vFloat c2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[2]);
    vFloat p = c4 * f + c3;
    vFloat c1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[1]);
    p = p * f + c2;
    vFloat c0 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[0]);
    p = p * f + c1;
    p = p * f + c0;
    vFloat y;
    y = setexp(p, ep);
    vFloat odd_origin_ax;
    vFloat negative_mantissa = -p;
    vFloat reciprocal = vConstFloatPrgm1 + vConstFloatPrgm0 * negative_mantissa;
    vInt inverse_exponent = ~ep;
    reciprocal = vConstFloatPrgm2 + reciprocal * negative_mantissa;
    inverse_exponent += 255;
    vFloat error = 1.0f + negative_mantissa * reciprocal;
    vFloat power = setexp(vFloat(1.0f), inverse_exponent);
    reciprocal = reciprocal + reciprocal * error;
    vFloat odd_origin_term;
    vFloat hyperbolic_inv = reciprocal * power;
    vFloat odd_origin_factor;
    y = compose_scale * hyperbolic_inv + y;
    return y;
}

template <typename Config>
inline vFloat hyperbolic_wh_terminal(vFloat x, vFloat y) {
    vFloat magnitude = x;
    v_if(magnitude >= __builtin_bit_cast(float, Config::kTerminalBoundBits)) {
        y = std::numeric_limits<float>::infinity();
    }
    v_endif;
    return y;
}
}  // namespace ckernel::sfpu::ttpoly::cosh

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_api.h"
#include "integrity_calls.hpp"

#if defined(TRISC_UNPACK)
constexpr uint32_t kProcessor = 2;
#elif defined(TRISC_MATH)
constexpr uint32_t kProcessor = 3;
#else
constexpr uint32_t kProcessor = 4;
#endif

void kernel_main() {
    for (uint32_t i = 0; i < N_ITERS; i++) {
        semantics_test::publish_group<kProcessor>(i);
    }
}

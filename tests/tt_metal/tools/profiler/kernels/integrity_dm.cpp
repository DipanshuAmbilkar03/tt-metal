// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "integrity_calls.hpp"

#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t kProcessor = 0;
#else
constexpr uint32_t kProcessor = 1;
#endif

void kernel_main() {
    for (uint32_t i = 0; i < N_ITERS; i++) {
        semantics_test::publish_group<kProcessor>(i);
    }
}

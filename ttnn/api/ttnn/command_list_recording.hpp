// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

// Kept light: device_operation.hpp includes this header, and almost every op includes device_operation.hpp.

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

namespace ttnn::experimental::detail {

// True while a CommandListBuilder records on the calling thread.
bool is_recording();

// Adds workload to the CommandListBuilder recording on the calling thread instead of enqueueing it.
void record_mesh_workload(
    std::string_view operation_name,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload);

}  // namespace ttnn::experimental::detail

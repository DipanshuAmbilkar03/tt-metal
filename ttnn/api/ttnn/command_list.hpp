// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <thread>

#include <tt-metalium/experimental/command_list.hpp>

#include "ttnn/common/queue_id.hpp"

// Experimental and subject to change. Shares unqualified names with tt::tt_metal::experimental; qualify the Metal ones.

namespace ttnn::experimental {

class CommandList;

// Records TTNN device operations into a command list instead of launching them.
// Only Metal 2.0 (ProgramSpec) programs can be recorded. Nothing needs to run before recording: recording compiles
// each workload and uploads its kernel binaries.
// Not movable: the recording thread holds a pointer to it.
class CommandListBuilder {
public:
    // Acquires Metal's builder reservation: one builder per MeshDevice.
    explicit CommandListBuilder(tt::tt_metal::distributed::MeshDevice& mesh_device);
    ~CommandListBuilder();

    CommandListBuilder(const CommandListBuilder&) = delete;
    CommandListBuilder& operator=(const CommandListBuilder&) = delete;
    CommandListBuilder(CommandListBuilder&&) = delete;
    CommandListBuilder& operator=(CommandListBuilder&&) = delete;

    // Device operations launched on the calling thread are recorded instead of enqueued until end_recording().
    // One recording builder per thread; one recording thread per builder.
    void begin_recording();
    void end_recording();
    bool is_recording() const;

    // Builds an independent command list bound to cq_id. Defaults to the calling thread's current command queue.
    // Repeatable; built lists outlive the builder.
    CommandList build(std::optional<QueueId> cq_id = std::nullopt) const;

    // Drops everything recorded so far. Not allowed while recording.
    void clear();

    // Releases the Metal reservation and invalidates the builder. Not allowed while recording.
    void deallocate();

    tt::tt_metal::distributed::MeshDevice& device() const;

    // Adds one workload to the recording.
    void record(tt::tt_metal::distributed::MeshWorkload& workload);

private:
    tt::tt_metal::experimental::CommandListBuilder metal_builder_;
    std::optional<std::thread::id> recording_thread_;
    bool valid_ = true;
};

// A replayable command list in device DRAM, bound to the command queue it was built for.
// Lists record raw device addresses: every tensor a recorded operation used must stay allocated while it replays.
class CommandList {
public:
    CommandList(CommandList&&) noexcept = default;
    CommandList& operator=(CommandList&&) noexcept = default;

    // Replays on the bound command queue.
    void enqueue(bool blocking = true, bool check_dram_overlap = false);

    // Releases device resources and invalidates the list. Repeated calls have no effect.
    void deallocate();

    tt::tt_metal::distributed::MeshDevice& device() const;
    QueueId cq_id() const;

private:
    friend class CommandListBuilder;
    explicit CommandList(tt::tt_metal::experimental::CommandList metal_list);

    tt::tt_metal::experimental::CommandList metal_list_;
};

// RAII begin_recording / end_recording for C++ callers.
class CommandListRecording {
public:
    explicit CommandListRecording(CommandListBuilder& builder) : builder_(builder) { builder_.begin_recording(); }
    ~CommandListRecording() { builder_.end_recording(); }

    CommandListRecording(const CommandListRecording&) = delete;
    CommandListRecording& operator=(const CommandListRecording&) = delete;

private:
    CommandListBuilder& builder_;
};

}  // namespace ttnn::experimental

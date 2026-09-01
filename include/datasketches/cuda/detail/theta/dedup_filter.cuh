/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#include <cuda/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda_runtime.h>

//! @file
//! @brief Block-local duplicate filter used by the Theta screen kernel.
//!
//! Every alternative implements the same three-function interface so that the
//! screen kernel is identical across variants and a benchmark comparing them
//! measures the filter and nothing else:
//!
//! - `filter_name`          a short label carried into benchmark output
//! - `filter_warp_collapse` whether the kernel runs `__match_any_sync` first
//! - `block_filter::init`   called by every thread, followed by a barrier
//! - `block_filter::admit`  called by every thread of a warp, `active` marks
//!                          the lanes actually holding a surviving hash, and
//!                          the return value says whether to emit the hash
//!
//! Every filter is best-effort. Missing a duplicate only means one extra entry
//! reaches the sort, which removes it anyway, so correctness never depends on
//! any of them. What no filter may ever do is drop a hash it has not proven to
//! be a duplicate.

//! @brief Slots in the block-local filter. Overridable to sweep filter size.
#ifndef DSCUDA_THETA_FILTER_SLOTS
#  define DSCUDA_THETA_FILTER_SLOTS 1024
#endif

#define DSCUDA_THETA_STRINGIFY_(x) #x
#define DSCUDA_THETA_STRINGIFY(x)  DSCUDA_THETA_STRINGIFY_(x)

namespace datasketches::cuda::detail::theta {

inline constexpr char filter_name[]        = "direct" DSCUDA_THETA_STRINGIFY(DSCUDA_THETA_FILTER_SLOTS);
inline constexpr bool filter_warp_collapse = true;

//! @brief Direct-mapped block-local duplicate cache.
//!
//! A slot holds a full 64-bit hash and a key is dropped only on an exact match,
//! so two hashes colliding on a slot merely evict each other and both are
//! emitted; a collision can never drop a distinct value. Being direct-mapped
//! rather than open-addressed is what keeps it safe: cost is one load and one
//! store no matter how full it is, with no probe chain to grow and no rehashing,
//! so an oversubscribed filter simply stops hitting instead of falling off a
//! cliff.
struct block_filter {
  static constexpr ::cuda::std::size_t slots = DSCUDA_THETA_FILTER_SLOTS;

  // Concurrent threads may race on a slot. Losing a remembered hash or emitting
  // a duplicate are both harmless, so the race is by design, but it is still a
  // race and the accesses are relaxed atomics rather than plain loads and stores.
  using cell_type = ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_block>;

  ::cuda::std::uint64_t table[slots];

  __device__ void init(unsigned int thread, unsigned int threads) noexcept
  {
    for (auto i = static_cast<::cuda::std::size_t>(thread); i < slots; i += threads) {
      table[i] = 0;
    }
  }

  [[nodiscard]] __device__ bool admit(::cuda::std::uint64_t hash, bool active) noexcept
  {
    if (!active) { return false; }
    cell_type cell{table[hash % slots]};
    if (cell.load(::cuda::memory_order_relaxed) == hash) { return false; }
    cell.store(hash, ::cuda::memory_order_relaxed);
    return true;
  }
};

}  // namespace datasketches::cuda::detail::theta

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
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/std/utility>

#include <cuda_runtime.h>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_ref_impl.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map_ref.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/types.cuh>

//! @file
//! @brief Block-local duplicate filter backed by a cuCollections open-addressing
//! table in shared memory.
//!
//! An exact set is the obvious way to deduplicate within a block: an insert both
//! tests membership and claims ownership, and unlike a direct-mapped cache two
//! hashes that collide do not evict each other. The reason this is not simply
//! better is that open addressing has a nonlinear occupancy curve. Cheap while
//! the table is empty, it degenerates into long probe chains as it fills, and a
//! table that is actually full makes every further distinct key walk every
//! bucket before failing.
//!
//! That failure mode is the whole problem, and it is worse than slow. This
//! CCCL revision's `fixed_capacity_map_ref` exposes `insert` returning a plain
//! `bool`, so a false result cannot distinguish "already present" from "table
//! full". Dropping every false loses distinct hashes and corrupts the sketch;
//! emitting every false gives up nearly all deduplication. There is no correct
//! one-pass filter to be had from the return value alone.
//!
//! The design here removes the ambiguity instead of trying to resolve it, by
//! never letting the table reach the state that produces it:
//!
//! - A shared counter tracks occupied slots. Above @ref load_limit the filter
//!   stops consulting the table and emits, so a probe never runs against a table
//!   that could be full and `insert == false` therefore always means duplicate.
//! - The counter is read before the insert and incremented after it, so it can
//!   only lag by the inserts currently in flight, which is bounded by the block
//!   width. With the limit at half of capacity the true load stays under 75%
//!   even in the worst interleaving, well inside the flat part of the curve.
//! - Degrading to pass-through rather than resetting is what keeps the cost
//!   proportional to the benefit. Input distinct enough to saturate the table is
//!   input with no duplicates to find, so the filter stops paying for a lookup
//!   that was not going to hit. Resetting instead would buy a fresh table for
//!   that same input at the price of a block-wide barrier per reset.
//! - Nothing here needs a barrier or a warp-collective operation, so the filter
//!   composes with the screen kernel's divergent leader selection unchanged.
//!
//! The Theta hash is already a well-mixed Murmur output truncated below theta,
//! so the table hashes with identity: its low bits are uniform, and re-mixing
//! them would only add latency to every probe.

//! @brief Slots in the table. Overridable to sweep capacity.
//!
//! 1024 slots of 16 bytes is 16 KiB per block, which still lets the full 2048
//! threads per SM resident on this architecture keep their blocks resident, so
//! the table costs no occupancy. Note that a slot is a key/payload pair even
//! though only the key is used: this revision of cuco offers a map and no set,
//! so the payload is dead weight the direct-mapped alternatives do not carry.
#ifndef DSCUDA_THETA_FILTER_SLOTS
#  define DSCUDA_THETA_FILTER_SLOTS 1024
#endif

//! @brief Slots per bucket, i.e. how many slots one probe step examines.
#ifndef DSCUDA_THETA_FILTER_BUCKET
#  define DSCUDA_THETA_FILTER_BUCKET 1
#endif

//! @brief Percent of capacity above which the filter stops consulting the table.
#ifndef DSCUDA_THETA_FILTER_LOAD_PCT
#  define DSCUDA_THETA_FILTER_LOAD_PCT 50
#endif

#define DSCUDA_THETA_STRINGIFY_(x) #x
#define DSCUDA_THETA_STRINGIFY(x)  DSCUDA_THETA_STRINGIFY_(x)

namespace datasketches::cuda::detail::theta {

namespace cuco = ::cuda::experimental::cuco;

inline constexpr char filter_name[] =
  "cucoset" DSCUDA_THETA_STRINGIFY(DSCUDA_THETA_FILTER_SLOTS) "b" DSCUDA_THETA_STRINGIFY(
    DSCUDA_THETA_FILTER_BUCKET) "l" DSCUDA_THETA_STRINGIFY(DSCUDA_THETA_FILTER_LOAD_PCT);
inline constexpr bool filter_warp_collapse = true;

//! @brief Hashes a Theta hash to itself.
struct identity_hash {
  using argument_type = ::cuda::std::uint64_t;
  using result_type   = ::cuda::std::uint64_t;

  [[nodiscard]] __host__ __device__ constexpr result_type operator()(
    argument_type key) const noexcept
  {
    return key;
  }
};

struct block_filter {
  using key_type   = ::cuda::std::uint64_t;
  using slot_type  = key_type;

  static constexpr int bucket_size           = DSCUDA_THETA_FILTER_BUCKET;
  using probing_scheme                       = cuco::linear_probing<1, identity_hash>;
  static constexpr ::cuda::std::size_t slots = DSCUDA_THETA_FILTER_SLOTS;

  static_assert(cuco::is_valid_capacity<probing_scheme, bucket_size>(slots),
                "filter slots must be a valid open-addressing capacity");

  using storage_ref_type =
    cuco::__open_addressing::__slot_storage_ref<slot_type, bucket_size, slots>;
  using set_ref_type = cuco::__open_addressing::__open_addressing_ref_impl<key_type,
                                                                          ::cuda::thread_scope_block,
                                                                          ::cuda::std::equal_to<key_type>,
                                                                          probing_scheme,
                                                                          storage_ref_type,
                                                                          false>;

  //! @brief Occupied slots above which the table is bypassed rather than probed.
  static constexpr int load_limit =
    static_cast<int>(slots * DSCUDA_THETA_FILTER_LOAD_PCT / 100);

  //! @brief Zero is never a valid Theta hash, so it is free as the empty sentinel.
  static constexpr key_type empty_key = 0;

  using counter_type = ::cuda::atomic_ref<int, ::cuda::thread_scope_block>;

  slot_type table[slots];
  int occupancy;

  [[nodiscard]] __device__ set_ref_type ref() noexcept
  {
    return set_ref_type{
      empty_key, ::cuda::std::equal_to<key_type>{}, probing_scheme{}, storage_ref_type{table, slots}};
  }

  __device__ void init(unsigned int thread, unsigned int threads) noexcept
  {
    for (auto i = static_cast<::cuda::std::size_t>(thread); i < slots; i += threads) {
      table[i] = empty_key;
    }
    if (thread == 0) { occupancy = 0; }
  }

  [[nodiscard]] __device__ bool admit(key_type hash, bool active) noexcept
  {
    if (!active) { return false; }

    // Above the limit the table is left alone entirely. This is both the cost
    // control and the correctness argument: no probe ever runs against a table
    // that might be full, so a failed insert below is unambiguously a duplicate.
    counter_type occupied{occupancy};
    if (occupied.load(::cuda::memory_order_relaxed) >= load_limit) { return true; }

    if (!ref().insert(hash)) { return false; }
    occupied.fetch_add(1, ::cuda::memory_order_relaxed);
    return true;
  }
};

}  // namespace datasketches::cuda::detail::theta

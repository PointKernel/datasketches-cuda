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

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <cuda_runtime.h>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/open_addressing_ref_impl.cuh>
#include <cuda/experimental/__cuco/detail/open_addressing/slot_storage_ref.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

//! @file
//! @brief Block-local duplicate filter backed by a cuCollections open-addressing
//! set in shared memory, retrieved in bulk rather than emitted per key.
//!
//! An exact set is the natural structure for this job: an insert both tests
//! membership and claims ownership, so unlike a direct-mapped cache two hashes
//! that collide do not evict each other. What makes open addressing dangerous is
//! that it degenerates as it fills, and a full table is worse than slow, because
//! `insert` returns a bare `bool` that cannot distinguish "already present" from
//! "no room left". Treating the second as the first silently drops distinct
//! hashes.
//!
//! Capacity is guaranteed rather than policed. A tile offers at most
//! @ref screen_tile_keys distinct hashes and the table holds twice that, so the
//! load can never exceed one half, a probe can never run against a full table,
//! and a failed insert is therefore unambiguously a duplicate. No occupancy
//! counter and no bypass path are needed. @ref init runs once per tile, which is
//! what keeps the invariant true under a grid-stride loop.
//!
//! Insertion is also the output. Survivors are not emitted as they are found;
//! the screen kernel retrieves every occupied slot once the tile is done, the
//! way `cuco::static_map::retrieve_all` does for a device-wide container. That
//! removes the per-key register staging, and it makes the emitted range exactly
//! deduplicated within the block rather than best-effort. The cost is that the
//! retrieval is proportional to capacity rather than to the hashes found, which
//! an insert returning the slot it claimed would fix; this revision of cuco does
//! not offer one.
//!
//! The slot is the key alone. The public `fixed_capacity_map_ref` stores a
//! key/payload pair that pads to 16 bytes and the payload is never read here, so
//! this uses the open-addressing implementation directly to get an 8-byte slot.
//! The Theta hash is already a well-mixed Murmur output truncated below theta,
//! so the table hashes with identity: re-mixing uniform bits only adds latency.

//! @brief Slots in the table. Must be at least twice @ref screen_tile_keys.
#ifndef DSCUDA_THETA_FILTER_SLOTS
#  define DSCUDA_THETA_FILTER_SLOTS 2048
#endif

//! @brief Slots per bucket, i.e. how many slots one probe step examines.
#ifndef DSCUDA_THETA_FILTER_BUCKET
#  define DSCUDA_THETA_FILTER_BUCKET 1
#endif

#define DSCUDA_THETA_STRINGIFY_(x) #x
#define DSCUDA_THETA_STRINGIFY(x)  DSCUDA_THETA_STRINGIFY_(x)

namespace datasketches::cuda::detail::theta {

namespace cuco = ::cuda::experimental::cuco;

inline constexpr char filter_name[] =
  "cucoset" DSCUDA_THETA_STRINGIFY(DSCUDA_THETA_FILTER_SLOTS) "b" DSCUDA_THETA_STRINGIFY(
    DSCUDA_THETA_FILTER_BUCKET);
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
  using key_type  = ::cuda::std::uint64_t;
  using slot_type = key_type;

  // `int` rather than a sized type because cuco's template parameter is `int`.
  static constexpr int bucket_size           = DSCUDA_THETA_FILTER_BUCKET;
  using probing_scheme                       = cuco::linear_probing<1, identity_hash>;
  static constexpr ::cuda::std::size_t slots = DSCUDA_THETA_FILTER_SLOTS;

  static_assert(cuco::is_valid_capacity<probing_scheme, bucket_size>(slots),
                "filter slots must be a valid open-addressing capacity");

  using storage_ref_type =
    cuco::__open_addressing::__slot_storage_ref<slot_type, bucket_size, slots>;
  using set_ref_type =
    cuco::__open_addressing::__open_addressing_ref_impl<key_type,
                                                        ::cuda::thread_scope_block,
                                                        ::cuda::std::equal_to<key_type>,
                                                        probing_scheme,
                                                        storage_ref_type,
                                                        false>;

  //! @brief Zero is never a valid Theta hash, so it is free as the empty sentinel.
  static constexpr key_type empty_key = 0;

  slot_type table[slots];

  [[nodiscard]] __device__ set_ref_type ref() noexcept
  {
    return set_ref_type{empty_key,
                        ::cuda::std::equal_to<key_type>{},
                        probing_scheme{},
                        storage_ref_type{table, slots}};
  }

  //! @brief Returns every slot to the empty sentinel. Called once per tile.
  __device__ void init(::cuda::std::uint32_t thread, ::cuda::std::uint32_t threads) noexcept
  {
    for (auto i = static_cast<::cuda::std::size_t>(thread); i < slots; i += threads) {
      table[i] = empty_key;
    }
  }

  //! @brief Inserts a surviving hash; the return value says whether it was new.
  //!
  //! Callers do not emit on the return value. A successful insert leaves the
  //! hash in the table for @ref screen_kernel to retrieve, and a failed one
  //! means an identical hash is already there and will be retrieved instead.
  [[nodiscard]] __device__ bool admit(key_type hash, bool active) noexcept
  {
    if (!active) { return false; }
    return ref().insert(hash);
  }
};

}  // namespace datasketches::cuda::detail::theta

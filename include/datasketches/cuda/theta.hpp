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

#include <cstddef>
#include <cstdint>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <utility>
#include <vector>

#include <datasketches/cuda/detail/theta/policy.cuh>
#include <datasketches/cuda/detail/theta/sketch_impl.hpp>

namespace datasketches::cuda {

//! @brief GPU Theta sketch with ordered compact serialization compatible with
//! the datasketches::compact_theta_sketch serialization version 3.
//!
//! Updates are batch-oriented. A single kernel hashes each key, screens it
//! against theta, and compacts the survivors; CUB radix sort, unique, and merge
//! primitives then fold those survivors into the retained set. An update that
//! begins with theta at its maximum is split internally so theta tightens
//! partway through the batch instead of after it. The retained hashes are always
//! ordered and trimmed to the smallest k = 2^lg_k values. Union (merge),
//! intersection, and A-not-B operate directly on those ordered device-resident
//! hashes.
//!
//! The current migration supports primitive device keys, uncompressed ordered
//! compact-v3 serialization, custom seeds, and p-sampling. It does not yet
//! support strings/byte spans, unordered or update-sketch wire images,
//! compressed v4 images, or legacy serialization versions.
//!
//! Every operation that changes storage synchronizes the supplied stream before
//! returning because the retained count determines the next allocation size.
//! The stream supplied at construction must remain alive until destruction;
//! retained buffers are rebound to that stream for stream-ordered deallocation.
//!
//! @tparam Key Primitive input key type.
//! @tparam MR Device-accessible memory resource type.
template <class Key, class MR = ::cuda::device_memory_pool_ref>
class theta_sketch {
 public:
  using key_type  = Key;
  using hash_type = std::uint64_t;

  static constexpr std::uint8_t default_lg_k  = detail::theta::default_lg_k;
  static constexpr std::uint64_t default_seed = detail::theta::default_seed;

  //! @brief Construct an empty sketch.
  theta_sketch(::cuda::stream_ref stream,
               MR mr,
               std::uint8_t lg_k  = default_lg_k,
               std::uint64_t seed = default_seed,
               float p            = 1.0F);

  theta_sketch(const theta_sketch&)            = delete;
  theta_sketch& operator=(const theta_sketch&) = delete;
  theta_sketch(theta_sketch&&)                 = default;
  theta_sketch& operator=(theta_sketch&&)      = default;
  ~theta_sketch()                              = default;

  //! @brief Hash, screen, sort, deduplicate, and merge a device range.
  template <class RandomAccessIt>
  void update(::cuda::stream_ref stream, RandomAccessIt first, RandomAccessIt last);

  //! @brief Replace this sketch with the union of this and other.
  template <class OtherMR>
  void merge(::cuda::stream_ref stream, const theta_sketch<Key, OtherMR>& other);

  //! @brief Replace this sketch with the intersection of this and other.
  template <class OtherMR>
  void intersect(::cuda::stream_ref stream, const theta_sketch<Key, OtherMR>& other);

  //! @brief Replace this sketch with the set difference this-minus-other.
  template <class OtherMR>
  void a_not_b(::cuda::stream_ref stream, const theta_sketch<Key, OtherMR>& other);

  //! @brief Restore the initial empty state.
  void reset(::cuda::stream_ref stream);

  [[nodiscard]] bool is_empty() const noexcept;
  [[nodiscard]] bool is_estimation_mode() const noexcept;
  [[nodiscard]] bool is_ordered() const noexcept;
  [[nodiscard]] std::uint8_t get_lg_k() const noexcept;
  [[nodiscard]] std::uint64_t get_theta64() const noexcept;
  [[nodiscard]] double get_theta() const noexcept;
  [[nodiscard]] std::uint16_t get_seed_hash() const noexcept;
  [[nodiscard]] std::size_t get_num_retained() const noexcept;
  [[nodiscard]] double get_estimate() const noexcept;
  [[nodiscard]] double get_lower_bound(std::uint8_t num_std_devs) const;
  [[nodiscard]] double get_upper_bound(std::uint8_t num_std_devs) const;

  //! @brief Copy the ordered retained hashes to host memory.
  [[nodiscard]] std::vector<hash_type> get_retained_hashes(::cuda::stream_ref stream) const;

  //! @brief Serialize as an ordered, uncompressed compact Theta v3 image.
  [[nodiscard]] std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref stream) const;

  //! @brief Deserialize an ordered, uncompressed compact Theta v3 image.
  //!
  //! Compact Theta images do not encode nominal k, so lg_k must be supplied
  //! by the caller when it differs from the default.
  static theta_sketch deserialize(::cuda::stream_ref stream,
                                  ::cuda::std::span<const std::uint8_t> bytes,
                                  MR mr,
                                  std::uint8_t lg_k  = default_lg_k,
                                  std::uint64_t seed = default_seed);

 private:
  template <class, class>
  friend class theta_sketch;

  detail::theta::sketch_impl<Key, MR> impl_;
};

}  // namespace datasketches::cuda

#include <datasketches/cuda/detail/theta/theta.inl>

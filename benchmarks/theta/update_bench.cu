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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <stdexcept>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <datasketches/cuda/detail/theta/dedup_filter.cuh>
#include <datasketches/cuda/theta.hpp>

#include <nvbench/nvbench.cuh>

namespace {

//! @brief Key types the update path is measured over.
//!
//! The Theta hash normalizes before hashing, so the key type changes how much
//! work happens per key before the 64-bit hash exists. Narrow integers and
//! floating point both take different paths through the normalizing hasher than
//! `uint64_t` does, and the screen kernel's memory traffic scales with the key
//! width, so a filter that looks free on 8-byte keys may not be on 4-byte ones.
using key_types = nvbench::type_list<std::uint32_t, std::uint64_t, float, double>;

//! @brief How duplicate keys are laid out in the input.
//!
//! Locality and duplicate count are separate axes and are not interchangeable.
//! Duplicate-heavy input keeps theta high for longer regardless of ordering,
//! which exercises the chunking path. Locality separately decides whether
//! repeats are visible to a kernel that only sees a bounded window of
//! consecutive keys, so a benchmark that only generates one arrival pattern
//! cannot distinguish an optimization that exploits locality from one that does
//! nothing.
enum class distribution : int {
  //! Every key distinct. No filter can win; measures pure filter overhead and
  //! how a table behaves as it saturates.
  unique,
  //! Few distinct values, repeats spaced `distinct` positions apart. Global
  //! duplicates with no block-local recurrence.
  scatter,
  //! Few distinct values in random order. Same duplicate count as `scatter`
  //! without its exact periodicity, which a direct-mapped table could alias on.
  shuffled,
  //! Repeats arrive in runs of 32, so duplicates are visible to a warp.
  warp_run,
  //! Repeats arrive in runs of one block tile, the sorted/grouped case.
  block_run,
  //! Power-law frequencies: a few values dominate, the tail is nearly unique.
  zipf
};

[[nodiscard]] distribution parse_distribution(const std::string& name)
{
  if (name == "unique") return distribution::unique;
  if (name == "scatter") return distribution::scatter;
  if (name == "shuffled") return distribution::shuffled;
  if (name == "warp_run") return distribution::warp_run;
  if (name == "block_run") return distribution::block_run;
  if (name == "zipf") return distribution::zipf;
  throw std::invalid_argument("unknown Dist value: " + name);
}

//! @brief Fraction of keys that are distinct, for every distribution but `unique`.
inline constexpr std::uint64_t duplicate_distinct_pct = 1;

//! @brief Keys occupying one run in the `warp_run` and `block_run` layouts.
inline constexpr std::uint64_t warp_run_length  = 32;
inline constexpr std::uint64_t block_run_length = 2048;

[[nodiscard]] __host__ __device__ std::uint64_t mix64(std::uint64_t x) noexcept
{
  x += 0x9e3779b97f4a7c15ULL;
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

[[nodiscard]] __host__ __device__ std::uint32_t mix32(std::uint32_t x) noexcept
{
  x ^= x >> 16;
  x *= 0x85ebca6bU;
  x ^= x >> 13;
  x *= 0xc2b2ae35U;
  x ^= x >> 16;
  return x;
}

//! @brief Maps an input position to a key with a chosen distinct count and layout.
//!
//! The position first becomes a distinct index in `[0, distinct)`, which fixes
//! the duplicate structure, and only then becomes a key. Both mixers are
//! bijections on their width, so distinct indices always produce distinct
//! integer keys and the requested distinct count is exact rather than
//! approximate. Floating-point keys take the distinct index directly, which is
//! representable exactly below 2^24 for `float` and 2^53 for `double`.
template <class Key>
struct generate_key {
  std::uint64_t distinct;
  distribution kind;

  [[nodiscard]] __host__ __device__ std::uint64_t distinct_index(
    std::uint64_t index) const noexcept
  {
    switch (kind) {
      case distribution::unique: return index;
      case distribution::scatter: return index % distinct;
      case distribution::shuffled: return mix64(index) % distinct;
      case distribution::warp_run: return (index / warp_run_length) % distinct;
      case distribution::block_run: return (index / block_run_length) % distinct;
      case distribution::zipf: {
        // Inverse-transform sampling of a power law over [0, distinct): a
        // uniform u maps to distinct^u, so the low indices take most of the mass
        // and the tail stays nearly unique.
        const double u = static_cast<double>(mix64(index) >> 11) * 0x1p-53;
        const auto d   = static_cast<std::uint64_t>(::pow(static_cast<double>(distinct), u));
        return d < distinct ? d : distinct - 1;
      }
    }
    return index;
  }

  [[nodiscard]] __host__ __device__ Key operator()(std::uint64_t index) const noexcept
  {
    const auto d = distinct_index(index);
    if constexpr (::cuda::std::is_floating_point_v<Key>) {
      return static_cast<Key>(d);
    } else if constexpr (sizeof(Key) <= 4) {
      return static_cast<Key>(mix32(static_cast<std::uint32_t>(d)));
    } else {
      return static_cast<Key>(mix64(d));
    }
  }
};

template <class Key>
thrust::device_vector<Key> make_keys(std::size_t count, distribution kind)
{
  const auto distinct = kind == distribution::unique
                          ? static_cast<std::uint64_t>(count)
                          : std::max<std::uint64_t>(1, count * duplicate_distinct_pct / 100);
  thrust::device_vector<Key> keys(count);
  thrust::transform(thrust::counting_iterator<std::uint64_t>(0),
                    thrust::counting_iterator<std::uint64_t>(count),
                    keys.begin(),
                    generate_key<Key>{distinct, kind});
  return keys;
}

//! @brief Cost of filling an empty sketch.
//!
//! A sketch entering update() with theta at its maximum rejects nothing, so this
//! measures the path where update() has to tighten theta partway through the
//! batch rather than screening against an already-small theta.
template <class Key>
void theta_update_cold(nvbench::state& state, nvbench::type_list<Key>)
{
  const auto num_keys = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto lg_k     = static_cast<std::uint8_t>(state.get_int64("LgK"));
  const auto kind     = parse_distribution(state.get_string("Dist"));

  const auto keys = make_keys<Key>(num_keys, kind);
  auto mr         = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  state.add_element_count(num_keys, "Keys");
  state.add_global_memory_reads<Key>(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    datasketches::cuda::theta_sketch<Key> sketch(stream, mr, lg_k);
    sketch.update(stream, keys.begin(), keys.end());
  });
}

//! @brief Cost of updating a sketch that has already reached its nominal k.
//!
//! This is the steady state for a streaming workload: theta is small, nearly
//! every key is rejected by the screen, and throughput is set by how fast the
//! sketch can hash and reject.
template <class Key>
void theta_update_warm(nvbench::state& state, nvbench::type_list<Key>)
{
  const auto num_keys = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto lg_k     = static_cast<std::uint8_t>(state.get_int64("LgK"));
  const auto kind     = parse_distribution(state.get_string("Dist"));

  const auto prime_keys = std::max<std::size_t>(1, num_keys / 10);
  const auto keys       = make_keys<Key>(num_keys, kind);
  const auto prime      = make_keys<Key>(prime_keys, distribution::unique);
  auto mr               = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  // Saturate the sketch outside the timed region. Re-running the measured update
  // against the same sketch is the steady state being measured: theta is already
  // small, so repeated updates neither grow the retained set nor move theta.
  ::cuda::stream setup_stream{::cuda::devices[0]};
  datasketches::cuda::theta_sketch<Key> sketch(setup_stream, mr, lg_k);
  sketch.update(setup_stream, prime.begin(), prime.end());
  setup_stream.sync();

  state.add_element_count(num_keys, "Keys");
  state.add_global_memory_reads<Key>(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    sketch.update(stream, keys.begin(), keys.end());
  });
}

//! @brief Cost of feeding a fixed key count through many small update() calls.
//!
//! Each call carries a fixed cost in launches and synchronizations that does not
//! shrink with the batch, so this separates per-call overhead from streaming
//! throughput. Sweeping Batch at a fixed Keys shows where the two cross over.
void theta_update_batched(nvbench::state& state)
{
  const auto num_keys = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto batch    = static_cast<std::size_t>(state.get_int64("Batch"));
  const auto lg_k     = static_cast<std::uint8_t>(state.get_int64("LgK"));

  const auto keys = make_keys<std::uint64_t>(num_keys, distribution::unique);
  auto mr         = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  state.add_element_count(num_keys, "Keys");
  state.add_global_memory_reads<std::uint64_t>(num_keys);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    datasketches::cuda::theta_sketch<std::uint64_t> sketch(stream, mr, lg_k);
    for (std::size_t offset = 0; offset < num_keys; offset += batch) {
      const auto end = std::min(offset + batch, num_keys);
      sketch.update(stream, keys.begin() + offset, keys.begin() + end);
    }
  });
}

//! @brief Cost of merging two saturated sketches.
void theta_merge(nvbench::state& state)
{
  const auto num_keys = static_cast<std::size_t>(state.get_int64("Keys"));
  const auto lg_k     = static_cast<std::uint8_t>(state.get_int64("LgK"));

  const auto left  = make_keys<std::uint64_t>(num_keys, distribution::unique);
  const auto right = make_keys<std::uint64_t>(num_keys, distribution::shuffled);
  auto mr          = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  // Both sides are built outside the timed region. Merging the same other sketch
  // repeatedly is stable once the first merge has run, so the measured call does
  // the same work on every iteration.
  ::cuda::stream setup_stream{::cuda::devices[0]};
  datasketches::cuda::theta_sketch<std::uint64_t> other(setup_stream, mr, lg_k);
  other.update(setup_stream, right.begin(), right.end());
  datasketches::cuda::theta_sketch<std::uint64_t> sketch(setup_stream, mr, lg_k);
  sketch.update(setup_stream, left.begin(), left.end());
  setup_stream.sync();

  state.add_element_count(std::size_t{1} << lg_k, "Retained");

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    ::cuda::stream_ref stream{launch.get_stream()};
    sketch.merge(stream, other);
  });
}

//! @brief Duplicate filter this binary was built with, carried into the output.
//!
//! Filter alternatives live on separate branches that differ only in
//! dedup_filter.cuh, so a comparison is assembled from the output of several
//! binaries. Tagging every row with the filter it came from is what makes those
//! outputs joinable without tracking which build directory produced which file.
const std::string filter_axis{datasketches::cuda::detail::theta::filter_name};

const std::vector<std::string> distributions{
  "unique", "scatter", "shuffled", "warp_run", "block_run", "zipf"};

}  // namespace

NVBENCH_BENCH_TYPES(theta_update_cold, NVBENCH_TYPE_AXES(key_types))
  .set_name("theta_update_cold")
  .set_type_axes_names({"KeyType"})
  .add_int64_axis("Keys", {1 << 24})
  .add_string_axis("Dist", distributions)
  .add_int64_axis("LgK", {12})
  .add_string_axis("Filter", {filter_axis});

NVBENCH_BENCH_TYPES(theta_update_warm, NVBENCH_TYPE_AXES(key_types))
  .set_name("theta_update_warm")
  .set_type_axes_names({"KeyType"})
  .add_int64_axis("Keys", {1 << 24})
  .add_string_axis("Dist", distributions)
  .add_int64_axis("LgK", {12})
  .add_string_axis("Filter", {filter_axis});

//! Scale sweep on the widest key type, where the sort the filter feeds is
//! largest and the chunking path matters most.
NVBENCH_BENCH_TYPES(theta_update_cold, NVBENCH_TYPE_AXES(nvbench::type_list<std::uint64_t>))
  .set_name("theta_update_cold_scale")
  .set_type_axes_names({"KeyType"})
  .add_int64_axis("Keys", {1 << 20, 1 << 24, 100'000'000})
  .add_string_axis("Dist", distributions)
  .add_int64_axis("LgK", {12, 20})
  .add_string_axis("Filter", {filter_axis});

NVBENCH_BENCH_TYPES(theta_update_warm, NVBENCH_TYPE_AXES(nvbench::type_list<std::uint64_t>))
  .set_name("theta_update_warm_scale")
  .set_type_axes_names({"KeyType"})
  .add_int64_axis("Keys", {1 << 20, 1 << 24, 100'000'000})
  .add_string_axis("Dist", distributions)
  .add_int64_axis("LgK", {12, 20})
  .add_string_axis("Filter", {filter_axis});

NVBENCH_BENCH(theta_update_batched)
  .set_name("theta_update_batched")
  .add_int64_axis("Keys", {1 << 24})
  .add_int64_axis("Batch", {1 << 16, 1 << 20, 1 << 24})
  .add_int64_axis("LgK", {12});

NVBENCH_BENCH(theta_merge)
  .set_name("theta_merge")
  .add_int64_axis("Keys", {1 << 22})
  .add_int64_axis("LgK", {12, 16});

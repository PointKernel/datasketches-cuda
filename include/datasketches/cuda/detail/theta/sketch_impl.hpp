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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda/memory_pool>
#include <cuda/std/functional>
#include <cuda/std/span>
#include <cuda/stream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <cub/device/device_merge.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/execution.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <datasketches/cuda/detail/common/error.hpp>
#include <datasketches/cuda/detail/theta/policy.cuh>
#include <datasketches/cuda/detail/theta/preamble.hpp>

#include <binomial_bounds.hpp>

namespace datasketches::cuda::detail::theta {

template <class Key, class MR = ::cuda::device_memory_pool_ref>
struct sketch_impl {
  using key_type          = Key;
  using hash_type         = std::uint64_t;
  using count_type        = std::uint64_t;
  using hash_buffer_type  = ::cuda::device_buffer<hash_type>;
  using count_buffer_type = ::cuda::device_buffer<count_type>;
  using env_type          = ::cuda::experimental::env_t<::cuda::mr::device_accessible>;

  struct buffer_result {
    hash_buffer_type data;
    std::size_t size;
  };

  std::uint8_t lg_k_;
  std::uint64_t seed_;
  float p_;
  std::uint64_t theta_;
  bool is_empty_;
  ::cuda::stream_ref allocation_stream_;
  MR mr_;
  hash_buffer_type hashes_;

  sketch_impl(::cuda::stream_ref stream, MR mr, std::uint8_t lg_k, std::uint64_t seed, float p)
    : lg_k_(check_lg_k_(lg_k)),
      seed_(seed),
      p_(check_p_(p)),
      theta_(starting_theta_(p_)),
      is_empty_(true),
      allocation_stream_(stream),
      mr_(std::move(mr)),
      hashes_(make_hash_buffer_(stream, 0))
  {
  }

  sketch_impl(const sketch_impl&)            = delete;
  sketch_impl& operator=(const sketch_impl&) = delete;
  sketch_impl(sketch_impl&&)                 = default;
  sketch_impl& operator=(sketch_impl&&)      = default;
  ~sketch_impl()                             = default;

  static std::uint8_t check_lg_k_(std::uint8_t lg_k)
  {
    if (lg_k < min_lg_k || lg_k > max_lg_k) {
      throw std::invalid_argument("theta_sketch lg_k must be in [5, 26]");
    }
    return lg_k;
  }

  static float check_p_(float p)
  {
    if (!(p > 0.0F && p <= 1.0F)) {
      throw std::invalid_argument("theta_sketch sampling probability must be in (0, 1]");
    }
    return p;
  }

  static std::uint64_t starting_theta_(float p)
  {
    return p < 1.0F ? static_cast<std::uint64_t>(static_cast<double>(max_theta) * p) : max_theta;
  }

  [[nodiscard]] std::size_t k_() const noexcept { return std::size_t{1} << lg_k_; }

  [[nodiscard]] std::uint64_t effective_theta_() const noexcept
  {
    return is_empty_ ? max_theta : theta_;
  }

  [[nodiscard]] env_type env_(::cuda::stream_ref stream) const { return env_type{mr_, stream}; }

  [[nodiscard]] hash_buffer_type make_hash_buffer_(::cuda::stream_ref stream,
                                                   std::size_t count) const
  {
    return ::cuda::make_buffer<hash_type, ::cuda::mr::device_accessible>(
      stream, mr_, count, ::cuda::no_init);
  }

  [[nodiscard]] count_buffer_type make_count_buffer_(::cuda::stream_ref stream) const
  {
    return ::cuda::make_buffer<count_type, ::cuda::mr::device_accessible>(
      stream, mr_, 1, count_type{0});
  }

  [[nodiscard]] static std::int64_t cub_count_(std::size_t count)
  {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::length_error("theta_sketch input exceeds CUB's signed 64-bit item count");
    }
    return static_cast<std::int64_t>(count);
  }

  [[nodiscard]] static std::size_t read_count_(::cuda::stream_ref stream,
                                               const count_buffer_type& count)
  {
    count_type host_count{};
    DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(
      &host_count, count.data(), sizeof(host_count), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    if (host_count > std::numeric_limits<std::size_t>::max()) {
      throw std::length_error("theta_sketch selected item count exceeds size_t");
    }
    return static_cast<std::size_t>(host_count);
  }

  [[nodiscard]] buffer_result select_(::cuda::stream_ref stream,
                                      const hash_type* input,
                                      std::size_t count,
                                      screen_hash predicate) const
  {
    auto output = make_hash_buffer_(stream, count);
    if (count == 0) return {std::move(output), 0};
    auto selected = make_count_buffer_(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::If(
      input, output.data(), selected.data(), cub_count_(count), predicate, env_(stream)));
    return {std::move(output), read_count_(stream, selected)};
  }

  [[nodiscard]] buffer_result select_(::cuda::stream_ref stream,
                                      const hash_type* input,
                                      std::size_t count,
                                      membership_filter predicate) const
  {
    auto output = make_hash_buffer_(stream, count);
    if (count == 0) return {std::move(output), 0};
    auto selected = make_count_buffer_(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::If(
      input, output.data(), selected.data(), cub_count_(count), predicate, env_(stream)));
    return {std::move(output), read_count_(stream, selected)};
  }

  [[nodiscard]] buffer_result unique_sorted_(::cuda::stream_ref stream,
                                             const hash_type* sorted,
                                             std::size_t count) const
  {
    auto output = make_hash_buffer_(stream, count);
    if (count == 0) return {std::move(output), 0};
    auto selected = make_count_buffer_(stream);
    DATASKETCHES_CUDA_TRY(cub::DeviceSelect::Unique(
      sorted, output.data(), selected.data(), cub_count_(count), env_(stream)));
    return {std::move(output), read_count_(stream, selected)};
  }

  [[nodiscard]] buffer_result sort_unique_(::cuda::stream_ref stream,
                                           hash_buffer_type&& input,
                                           std::size_t count) const
  {
    if (count == 0) return {std::move(input), 0};
    auto alternate = make_hash_buffer_(stream, count);
    cub::DoubleBuffer<hash_type> keys(input.data(), alternate.data());
    DATASKETCHES_CUDA_TRY(
      cub::DeviceRadixSort::SortKeys(keys, cub_count_(count), 0, 63, env_(stream)));
    return unique_sorted_(stream, keys.Current(), count);
  }

  [[nodiscard]] buffer_result merge_unique_(::cuda::stream_ref stream,
                                            const hash_type* first,
                                            std::size_t first_size,
                                            const hash_type* second,
                                            std::size_t second_size) const
  {
    const std::size_t merged_size = first_size + second_size;
    auto merged                   = make_hash_buffer_(stream, merged_size);
    if (merged_size == 0) return {std::move(merged), 0};

    if (first_size == 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(merged.data(),
                                            second,
                                            second_size * sizeof(hash_type),
                                            cudaMemcpyDeviceToDevice,
                                            stream.get()));
    } else if (second_size == 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(merged.data(),
                                            first,
                                            first_size * sizeof(hash_type),
                                            cudaMemcpyDeviceToDevice,
                                            stream.get()));
    } else {
      DATASKETCHES_CUDA_TRY(cub::DeviceMerge::MergeKeys(first,
                                                        cub_count_(first_size),
                                                        second,
                                                        cub_count_(second_size),
                                                        merged.data(),
                                                        ::cuda::std::less<>{},
                                                        env_(stream)));
    }
    return unique_sorted_(stream, merged.data(), merged_size);
  }

  [[nodiscard]] hash_buffer_type copy_prefix_(::cuda::stream_ref stream,
                                              const hash_type* input,
                                              std::size_t count) const
  {
    auto output = make_hash_buffer_(stream, count);
    if (count != 0) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(
        output.data(), input, count * sizeof(hash_type), cudaMemcpyDeviceToDevice, stream.get()));
    }
    stream.sync();
    output.set_stream(allocation_stream_);
    return output;
  }

  void install_(::cuda::stream_ref stream,
                const hash_type* input,
                std::size_t count,
                std::uint64_t theta,
                bool empty,
                bool trim)
  {
    std::size_t retained = count;
    if (trim && count > k_()) {
      DATASKETCHES_CUDA_TRY(
        cudaMemcpyAsync(&theta, input + k_(), sizeof(theta), cudaMemcpyDeviceToHost, stream.get()));
      stream.sync();
      retained = k_();
    }
    auto next = copy_prefix_(stream, input, retained);
    hashes_   = std::move(next);
    theta_    = empty ? starting_theta_(p_) : theta;
    is_empty_ = empty;
  }

  void set_empty_(::cuda::stream_ref stream)
  {
    auto empty = make_hash_buffer_(stream, 0);
    stream.sync();
    empty.set_stream(allocation_stream_);
    hashes_   = std::move(empty);
    theta_    = starting_theta_(p_);
    is_empty_ = true;
  }

  template <class RandomAccessIt>
  void update(::cuda::stream_ref stream, RandomAccessIt first, RandomAccessIt last)
  {
    const auto distance = last - first;
    if (distance < 0) {
      throw std::invalid_argument("theta_sketch::update requires a non-negative range");
    }
    const auto count = static_cast<std::size_t>(distance);
    if (count == 0) return;

    is_empty_ = false;

    auto hashed = make_hash_buffer_(stream, count);
    DATASKETCHES_CUDA_TRY(cub::DeviceTransform::Transform(
      first, hashed.data(), cub_count_(count), theta_hash<Key>{seed_}, env_(stream)));

    auto screened = select_(stream, hashed.data(), count, screen_hash{theta_});
    if (screened.size == 0) return;

    auto incoming = sort_unique_(stream, std::move(screened.data), screened.size);
    auto combined =
      merge_unique_(stream, hashes_.data(), hashes_.size(), incoming.data.data(), incoming.size);
    install_(stream, combined.data.data(), combined.size, theta_, false, true);
  }

  template <class OtherMR>
  void merge(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    if (other.is_empty_) return;
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument("theta_sketch::merge: seed hash mismatch");
    }

    const std::uint64_t theta = std::min(theta_, other.effective_theta_());
    auto combined             = merge_unique_(
      stream, hashes_.data(), hashes_.size(), other.hashes_.data(), other.hashes_.size());
    auto screened = select_(stream, combined.data.data(), combined.size, screen_hash{theta});
    install_(stream, screened.data.data(), screened.size, theta, false, true);
  }

  template <class OtherMR>
  void intersect(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    if (is_empty_) return;
    if (other.is_empty_) {
      set_empty_(stream);
      return;
    }
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument("theta_sketch::intersect: seed hash mismatch");
    }

    const std::uint64_t theta = std::min(effective_theta_(), other.effective_theta_());
    auto result =
      select_(stream,
              hashes_.data(),
              hashes_.size(),
              membership_filter{other.hashes_.data(), other.hashes_.size(), theta, true});
    const bool empty = result.size == 0 && theta == max_theta;
    install_(stream, result.data.data(), result.size, theta, empty, false);
  }

  template <class OtherMR>
  void a_not_b(::cuda::stream_ref stream, const sketch_impl<Key, OtherMR>& other)
  {
    if (is_empty_ || (!hashes_.empty() && other.is_empty_)) return;
    if (::compute_seed_hash(seed_) != ::compute_seed_hash(other.seed_)) {
      throw std::invalid_argument("theta_sketch::a_not_b: seed hash mismatch");
    }

    const std::uint64_t theta = std::min(effective_theta_(), other.effective_theta_());
    auto result =
      select_(stream,
              hashes_.data(),
              hashes_.size(),
              membership_filter{other.hashes_.data(), other.hashes_.size(), theta, false});
    const bool empty = result.size == 0 && theta == max_theta;
    install_(stream, result.data.data(), result.size, theta, empty, false);
  }

  void reset(::cuda::stream_ref stream) { set_empty_(stream); }

  [[nodiscard]] bool is_empty() const noexcept { return is_empty_; }

  [[nodiscard]] bool is_estimation_mode() const noexcept
  {
    return !is_empty_ && theta_ < max_theta;
  }

  [[nodiscard]] std::uint8_t get_lg_k() const noexcept { return lg_k_; }

  [[nodiscard]] std::uint64_t get_theta64() const noexcept { return effective_theta_(); }

  [[nodiscard]] double get_theta() const noexcept
  {
    return static_cast<double>(effective_theta_()) / static_cast<double>(max_theta);
  }

  [[nodiscard]] std::uint16_t get_seed_hash() const noexcept { return ::compute_seed_hash(seed_); }

  [[nodiscard]] std::size_t get_num_retained() const noexcept { return hashes_.size(); }

  [[nodiscard]] double get_estimate() const noexcept
  {
    return static_cast<double>(hashes_.size()) / get_theta();
  }

  [[nodiscard]] double get_lower_bound(std::uint8_t num_std_devs) const
  {
    if (!is_estimation_mode()) return static_cast<double>(hashes_.size());
    return ::datasketches::binomial_bounds::get_lower_bound(
      hashes_.size(), get_theta(), num_std_devs);
  }

  [[nodiscard]] double get_upper_bound(std::uint8_t num_std_devs) const
  {
    if (!is_estimation_mode()) return static_cast<double>(hashes_.size());
    return ::datasketches::binomial_bounds::get_upper_bound(
      hashes_.size(), get_theta(), num_std_devs);
  }

  [[nodiscard]] std::vector<hash_type> get_retained_hashes(::cuda::stream_ref stream) const
  {
    std::vector<hash_type> entries(hashes_.size());
    if (!entries.empty()) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(entries.data(),
                                            hashes_.data(),
                                            entries.size() * sizeof(hash_type),
                                            cudaMemcpyDeviceToHost,
                                            stream.get()));
    }
    stream.sync();
    return entries;
  }

  [[nodiscard]] std::vector<std::uint8_t> serialize_compact(::cuda::stream_ref stream) const
  {
    return serialize_compact_v3(
      is_empty_, get_seed_hash(), effective_theta_(), get_retained_hashes(stream));
  }

  void load_compact_(::cuda::stream_ref stream, const compact_image& image)
  {
    if (image.entries.size() > k_()) {
      throw std::invalid_argument(
        "theta_sketch::deserialize: retained entries exceed configured nominal k");
    }
    auto next = make_hash_buffer_(stream, image.entries.size());
    if (!image.entries.empty()) {
      DATASKETCHES_CUDA_TRY(cudaMemcpyAsync(next.data(),
                                            image.entries.data(),
                                            image.entries.size() * sizeof(hash_type),
                                            cudaMemcpyHostToDevice,
                                            stream.get()));
    }
    stream.sync();
    next.set_stream(allocation_stream_);
    hashes_   = std::move(next);
    theta_    = image.empty ? starting_theta_(p_) : image.theta;
    is_empty_ = image.empty;
  }
};

}  // namespace datasketches::cuda::detail::theta

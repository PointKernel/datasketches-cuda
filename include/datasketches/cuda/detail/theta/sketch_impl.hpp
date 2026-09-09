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

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/execution.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <datasketches/cuda/detail/common/error.hpp>
#include <datasketches/cuda/detail/theta/policy.cuh>
#include <datasketches/cuda/detail/theta/preamble.hpp>
#include <datasketches/cuda/detail/theta/screen.cuh>

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

  //! @brief Outcome of a bounded screen pass.
  //!
  //! The kernel counts every survivor but writes only as many as fit. When
  //! @ref overflowed is set, @ref data holds an arbitrary subset of the
  //! survivors and must be discarded; @ref count is still exact.
  struct screen_result {
    hash_buffer_type data;
    std::size_t count;
    bool overflowed;
  };

  std::uint8_t lg_k_;
  std::uint64_t seed_;
  float p_;
  std::uint64_t theta_;
  bool is_empty_;
  //! @brief Screen output capacity in hashes; see @ref survivor_budget_.
  std::size_t survivor_budget_hashes_;
  //! @brief Number of screen passes that overflowed and were retried. A
  //! diagnostic for tests; it never influences a result.
  std::size_t screen_overflows_;
  ::cuda::stream_ref allocation_stream_;
  MR mr_;
  hash_buffer_type hashes_;

  sketch_impl(::cuda::stream_ref stream, MR mr, std::uint8_t lg_k, std::uint64_t seed, float p)
    : lg_k_(check_lg_k_(lg_k)),
      seed_(seed),
      p_(check_p_(p)),
      theta_(starting_theta_(p_)),
      is_empty_(true),
      survivor_budget_hashes_(default_survivor_budget_()),
      screen_overflows_(0),
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

  //! @brief Highest bit position that a hash below @p theta can occupy.
  //!
  //! Radix sort runs one pass per fixed number of bits, so bounding the key range
  //! by theta drops whole passes once the sketch has left the initial
  //! theta == max_theta state.
  [[nodiscard]] static int significant_bits_(std::uint64_t theta) noexcept
  {
    int bits = 0;
    while (theta != 0) {
      ++bits;
      theta >>= 1;
    }
    return bits == 0 ? 1 : bits;
  }

  [[nodiscard]] buffer_result sort_unique_(::cuda::stream_ref stream,
                                           hash_buffer_type&& input,
                                           std::size_t count,
                                           std::uint64_t bound) const
  {
    if (count == 0) return {std::move(input), 0};
    auto alternate = make_hash_buffer_(stream, count);
    cub::DoubleBuffer<hash_type> keys(input.data(), alternate.data());
    DATASKETCHES_CUDA_TRY(cub::DeviceRadixSort::SortKeys(
      keys, cub_count_(count), 0, significant_bits_(bound), env_(stream)));
    return unique_sorted_(stream, keys.Current(), count);
  }

  //! @brief Hashes, screens, and compacts a key range in one pass.
  //!
  //! The output buffer holds `min(count, survivor_budget_())` hashes rather
  //! than one per key, so the scratch memory of an update is bounded by the
  //! budget instead of by the batch. The kernel counts survivors exactly and
  //! drops the writes that do not fit; the caller checks @ref
  //! screen_result::overflowed and retries with a smaller chunk when that
  //! happens. It cannot happen for a chunk of at most the budget, so a retry
  //! loop that halves the chunk always terminates.
  template <class RandomAccessIt>
  [[nodiscard]] screen_result screen_(::cuda::stream_ref stream,
                                      RandomAccessIt first,
                                      std::size_t count) const
  {
    const std::size_t capacity = std::min(count, survivor_budget_());
    auto output                = make_hash_buffer_(stream, capacity);
    if (count == 0) return {std::move(output), 0, false};
    auto selected = make_count_buffer_(stream);
    screen_kernel<<<screen_grid_size(count), screen_block_threads, 0, stream.get()>>>(
      first,
      count,
      theta_hash<Key>{seed_},
      theta_,
      output.data(),
      capacity,
      reinterpret_cast<unsigned long long*>(selected.data()));
    DATASKETCHES_CUDA_TRY(cudaGetLastError());
    const std::size_t survivors = read_count_(stream, selected);
    return {std::move(output), survivors, survivors > capacity};
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

  //! @brief Multiple of k a first chunk aims to cover.
  //!
  //! With theta at its maximum every key survives, so a chunk of this many keys
  //! yields that many candidates for a sketch that keeps k. A few multiples of k
  //! is enough to drive theta below its maximum whenever the batch holds at
  //! least k distinct keys, while keeping the sort that chunk pays for small.
  static constexpr std::size_t chunk_target_multiple = 16;

  //! @brief Smallest chunk worth a launch.
  //!
  //! A chunk costs a kernel launch and a synchronization regardless of its size,
  //! so very small k values should not produce correspondingly tiny chunks.
  static constexpr std::size_t min_chunk_keys = std::size_t{1} << 20;

  //! @brief Largest factor by which one chunk may exceed the one before it.
  //!
  //! The survivor estimate that sizes a chunk partly rests on the duplicate
  //! factor the previous pass showed, and input can change character. A pass
  //! that overflows is screened again for nothing, so the growth cap keeps that
  //! wasted pass proportional to the work already done, while still letting a
  //! batch of any size be covered in a handful of chunks.
  static constexpr std::size_t max_chunk_growth = 64;

  //! @brief Margin between a chunk's expected survivors and the budget.
  //!
  //! Survivors of distinct keys are a binomial draw at the theta fraction, and
  //! at millions of expected survivors ten percent is hundreds of standard
  //! deviations, so a chunk sized this way cannot overflow on its own.
  static constexpr double budget_margin = 1.1;

  //! @brief Hedge applied to the measured duplicate factor.
  //!
  //! The duplicate factor is measured on the previous pass and only holds if
  //! the input keeps its character. Doubling it before use means the screen can
  //! overflow only if the share of keys getting past the duplicate filter more
  //! than doubles from one chunk to the next.
  static constexpr double duplicate_hedge = 2.0;

  //! @brief Smallest survivor budget, in hashes, that any k is given.
  //!
  //! 2^22 hashes is 32 MiB of screen output. Below that the per-chunk fixed
  //! costs (launch, readback, CUB temporary storage) start to show against the
  //! streaming work, and above it the scratch footprint stops being negligible
  //! next to the retained k entries for the lg_k values in common use.
  static constexpr std::size_t min_survivor_budget = std::size_t{1} << 22;

  //! @brief Default number of survivors a single screen pass may produce.
  //!
  //! The screen writes its survivors to a buffer that has to be allocated before
  //! their count is known. Sizing it from the key count keeps the scratch memory
  //! of an update proportional to the batch, which for a batch of 2^31 keys is
  //! 16 GiB even though the survivors number about count times the pass rate.
  //! Sizing it from a fixed budget instead, and cutting the batch into chunks
  //! whose expected survivors fit, bounds scratch memory by the budget for any
  //! batch. 16 k is already what the first chunk at maximum theta aims to
  //! produce, so the budget is that or @ref min_survivor_budget, whichever is
  //! larger; the minimum prevents small k from forcing tiny chunks.
  [[nodiscard]] std::size_t default_survivor_budget_() const noexcept
  {
    return std::max(min_survivor_budget, chunk_target_multiple * k_());
  }

  //! @brief Number of survivors a single screen pass may produce.
  //!
  //! Bounds the output buffer of @ref screen_ and, through @ref next_chunk_,
  //! the scratch memory of @ref update. Defaults to @ref
  //! default_survivor_budget_ and is only changed by tests, which shrink it to
  //! force the overflow path.
  [[nodiscard]] std::size_t survivor_budget_() const noexcept { return survivor_budget_hashes_; }

  //! @brief Overrides the survivor budget. Intended for tests.
  //!
  //! Any positive budget yields the same result; a smaller one only costs more
  //! chunks and, below the natural chunk sizes, overflow retries.
  void set_survivor_budget_(std::size_t hashes)
  {
    if (hashes == 0) {
      throw std::invalid_argument("theta_sketch survivor budget must be positive");
    }
    survivor_budget_hashes_ = hashes;
  }

  //! @brief Size of the next update chunk.
  //!
  //! Entering an update with theta at its maximum means no distinct key is
  //! rejected, so a single pass over a large batch would sort the whole batch
  //! even though the sketch keeps only k entries. Splitting lets theta tighten
  //! partway through, exactly as the CPU sketch does on every insert, after
  //! which the remaining keys are screened rather than sorted. Chunks double
  //! while theta does stay at its maximum, which bounds the pass count
  //! logarithmically for a batch that holds fewer than k distinct keys.
  //!
  //! Every chunk is also sized so that its expected survivors fit the survivor
  //! budget with @ref budget_margin to spare, which is what bounds the screen
  //! buffer. The expected survivor fraction is theta over the hash space, exact
  //! for distinct keys, times the duplicate factor: the share of keys the
  //! screen's duplicate filter lets through, which the previous pass measured
  //! and which is orders of magnitude below one for grouped or repeated input.
  //! Sizing from theta alone would cut such a batch into dozens of small chunks
  //! that each pay a launch and a readback. The measured factor is hedged by
  //! @ref duplicate_hedge before use, so an overflow takes a change in the
  //! input's character; when the screen does overflow, nothing is installed and
  //! the caller retries with a chunk sized from the exact survivor count of the
  //! failed pass. That failed pass is wasted work, so whenever the estimate
  //! leans on a duplicate factor small enough to make an overflow possible at
  //! all, growth per chunk is capped at @ref max_chunk_growth to keep the waste
  //! proportional to the work already done. Below maximum theta the chunk is
  //! never smaller than @ref min_chunk_keys, which is within the default budget
  //! so that floor cannot overflow either.
  //!
  //! @param remaining Keys of the batch not yet installed
  //! @param previous Size of the last installed chunk, zero at the start
  //! @param duplicate_factor Survivors per key of the last screen pass divided
  //!   by the theta fraction that pass was screened at, nominally in [0, 1], or
  //!   negative if no pass has run yet in this update
  [[nodiscard]] std::size_t next_chunk_(std::size_t remaining,
                                        std::size_t previous,
                                        double duplicate_factor) const noexcept
  {
    const std::size_t budget = survivor_budget_();
    const double theta_rate  = static_cast<double>(theta_) / static_cast<double>(max_theta);
    const double hedged =
      duplicate_factor >= 0.0 ? std::min(1.0, duplicate_hedge * duplicate_factor) : 1.0;
    const double rate   = theta_rate * hedged;
    const double target = rate > 0.0
                            ? std::floor(static_cast<double>(budget) / (budget_margin * rate))
                            : static_cast<double>(remaining);
    std::size_t sized =
      target >= static_cast<double>(remaining) ? remaining : static_cast<std::size_t>(target);
    if (hedged < 1.0 && previous != 0 && sized / max_chunk_growth > previous) {
      sized = previous * max_chunk_growth;
    }
    sized = std::max<std::size_t>(sized, 1);
    if (theta_ == max_theta) {
      const auto doubling = std::max({chunk_target_multiple * k_(), min_chunk_keys, previous * 2});
      return std::min({doubling, sized, remaining});
    }
    return std::min(std::max(sized, min_chunk_keys), remaining);
  }

  //! @brief Outcome of folding one chunk into the sketch.
  struct chunk_outcome {
    //! True if the chunk was installed. False if the screen overflowed its
    //! buffer, in which case nothing was installed and the sketch state is
    //! unchanged; the caller must retry the same keys with a smaller chunk.
    bool installed;
    //! Exact number of survivors the screen counted, installed or not.
    std::size_t survivors;
  };

  //! @brief Folds one chunk of keys into the sketch.
  template <class RandomAccessIt>
  [[nodiscard]] chunk_outcome update_chunk_(::cuda::stream_ref stream,
                                            RandomAccessIt first,
                                            std::size_t count)
  {
    auto screened = screen_(stream, first, count);
    if (screened.overflowed) { return {false, screened.count}; }
    if (screened.count == 0) { return {true, 0}; }

    auto incoming = sort_unique_(stream, std::move(screened.data), screened.count, theta_);
    auto combined =
      merge_unique_(stream, hashes_.data(), hashes_.size(), incoming.data.data(), incoming.size);
    install_(stream, combined.data.data(), combined.size, theta_, false, true);
    return {true, screened.count};
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

    // Set up front rather than per chunk: an overflow retry leaves theta_ and
    // hashes_ untouched, and a chunk whose keys are all rejected still counts
    // as having been seen, so nothing below depends on this flag.
    is_empty_ = false;

    std::size_t offset      = 0;
    std::size_t previous    = 0;
    double duplicate_factor = -1.0;
    while (offset < count) {
      auto chunk = next_chunk_(count - offset, previous, duplicate_factor);
      // Theta is fixed until something is installed, so a pass's survivors per
      // key over the theta fraction it was screened at is its duplicate factor.
      const double theta_rate = static_cast<double>(theta_) / static_cast<double>(max_theta);
      auto outcome            = update_chunk_(stream, first + offset, chunk);
      while (!outcome.installed) {
        ++screen_overflows_;
        duplicate_factor =
          static_cast<double>(outcome.survivors) / static_cast<double>(chunk) / theta_rate;
        // The exact survivor count of the failed pass usually lands the retry
        // within the budget at once; halving guarantees progress regardless,
        // since a chunk of at most the budget cannot overflow.
        chunk = std::max<std::size_t>(
          1, std::min(chunk / 2, next_chunk_(count - offset, previous, duplicate_factor)));
        outcome = update_chunk_(stream, first + offset, chunk);
      }
      duplicate_factor =
        static_cast<double>(outcome.survivors) / static_cast<double>(chunk) / theta_rate;
      offset += chunk;
      previous = chunk;
    }
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

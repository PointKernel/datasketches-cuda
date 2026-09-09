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
#include <cuda/std/span>
#include <cuda/stream>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <datasketches/cuda/theta.hpp>

#include "cpu_reference.hpp"

namespace {

std::vector<std::uint64_t> make_keys(std::size_t count, std::uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::vector<std::uint64_t> keys(count);
  for (auto& key : keys)
    key = rng();
  return keys;
}

}  // namespace

TEST_CASE("Theta starts empty and exact", "[theta][basic]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> sketch(stream, mr, 12);

  REQUIRE(sketch.is_empty());
  REQUIRE(sketch.is_ordered());
  REQUIRE_FALSE(sketch.is_estimation_mode());
  REQUIRE(sketch.get_lg_k() == 12);
  REQUIRE(sketch.get_num_retained() == 0);
  REQUIRE(sketch.get_estimate() == 0.0);
  REQUIRE(sketch.get_theta() == 1.0);

  const auto bytes = sketch.serialize_compact(stream);
  REQUIRE(bytes == theta_test::cpu_image(std::vector<std::uint64_t>{}, 12, 9001));
  const auto cpu = theta_test::cpu_metadata(bytes);
  REQUIRE(cpu.empty);
  REQUIRE(cpu.ordered);
}

TEST_CASE("Theta compact v3 bytes match CPU in exact mode", "[theta][parity][serialization]")
{
  constexpr std::uint8_t lg_k  = 12;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(1000, 0x12345678ULL);
  keys.insert(keys.end(), keys.begin(), keys.begin() + 200);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
  REQUIRE(gpu.get_num_retained() == 1000);
  REQUIRE(gpu.get_estimate() == 1000.0);
}

TEST_CASE("Theta compact v3 bytes match trimmed CPU in estimation mode",
          "[theta][parity][serialization]")
{
  constexpr std::uint8_t lg_k                      = 10;
  constexpr std::uint64_t seed                     = 123456789;
  auto keys                                        = make_keys(100000, 0xabcdefULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == (std::size_t{1} << lg_k));
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));

  const double estimate = gpu.get_estimate();
  const double lower    = gpu.get_lower_bound(2);
  const double upper    = gpu.get_upper_bound(2);
  REQUIRE(lower <= estimate);
  REQUIRE(estimate <= upper);
}

TEST_CASE("Theta incremental batches match a single CPU update sketch",
          "[theta][parity][incremental]")
{
  constexpr std::uint8_t lg_k                      = 10;
  auto keys                                        = make_keys(50000, 0x31415926ULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k);
  gpu.update(stream, device_keys.begin(), device_keys.begin() + 7777);
  gpu.update(stream, device_keys.begin() + 7777, device_keys.begin() + 23456);
  gpu.update(stream, device_keys.begin() + 23456, device_keys.end());

  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, 9001));
}

TEST_CASE("Theta p-sampling and custom seed match CPU", "[theta][parity][sampling]")
{
  constexpr std::uint8_t lg_k                      = 12;
  constexpr std::uint64_t seed                     = 0xdecafbadULL;
  constexpr float p                                = 0.125F;
  auto keys                                        = make_keys(10000, 0xf00dULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed, p);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed, p));
}

TEST_CASE("Theta compact v3 round trips through CPU and GPU", "[theta][serialization]")
{
  constexpr std::uint8_t lg_k                      = 10;
  auto keys                                        = make_keys(10000, 0xfeedULL);
  thrust::device_vector<std::uint64_t> device_keys = keys;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> source(stream, mr, lg_k);
  source.update(stream, device_keys.begin(), device_keys.end());
  const auto bytes = source.serialize_compact(stream);

  const auto cpu = theta_test::cpu_metadata(bytes);
  REQUIRE(cpu.retained == source.get_num_retained());
  REQUIRE(cpu.theta == source.get_theta64());

  auto restored = datasketches::cuda::theta_sketch<std::uint64_t>::deserialize(
    stream, ::cuda::std::span<const std::uint8_t>{bytes.data(), bytes.size()}, mr, lg_k);
  REQUIRE(restored.serialize_compact(stream) == bytes);
  REQUIRE(restored.get_estimate() == Catch::Approx(source.get_estimate()));
}

TEST_CASE("Theta validates constructor and compact image", "[theta][validation]")
{
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  REQUIRE_THROWS_AS((datasketches::cuda::theta_sketch<std::uint64_t>(stream, mr, 4)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((datasketches::cuda::theta_sketch<std::uint64_t>(stream, mr, 12, 9001, 0.0F)),
                    std::invalid_argument);

  std::vector<std::uint8_t> invalid(8, 0);
  invalid[0] = 1;
  invalid[1] = 4;
  invalid[2] = 3;
  REQUIRE_THROWS_AS(
    datasketches::cuda::theta_sketch<std::uint64_t>::deserialize(
      stream, ::cuda::std::span<const std::uint8_t>{invalid.data(), invalid.size()}, mr),
    std::invalid_argument);

  std::vector<std::uint8_t> truncated(8, 0);
  truncated[0] = 3;
  truncated[1] = 3;
  truncated[2] = 3;
  truncated[5] = (1U << 1) | (1U << 3) | (1U << 4);
  datasketches::cuda::theta_sketch<std::uint64_t> seed_source(stream, mr);
  const auto seed_hash = seed_source.get_seed_hash();
  truncated[6]         = static_cast<std::uint8_t>(seed_hash);
  truncated[7]         = static_cast<std::uint8_t>(seed_hash >> 8);
  REQUIRE_THROWS_AS(
    datasketches::cuda::theta_sketch<std::uint64_t>::deserialize(
      stream, ::cuda::std::span<const std::uint8_t>{truncated.data(), truncated.size()}, mr),
    std::invalid_argument);
}

TEST_CASE("Theta multi-chunk update matches a single CPU sketch", "[theta][parity][chunking]")
{
  // Large enough that update() splits the batch internally: the sketch enters
  // with theta at its maximum, so the first chunk is bounded and later chunks
  // run against a tightened theta. Parity with the CPU sketch must not depend
  // on how the batch happens to be split.
  constexpr std::uint8_t lg_k  = 12;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(5'000'000, 0xc0ffeeULL);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  datasketches::cuda::theta_sketch<std::uint64_t> gpu(stream, mr, lg_k, seed);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == (std::size_t{1} << lg_k));
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta batch splitting does not change the result", "[theta][chunking]")
{
  // The same keys fed as one call and as several must produce identical images,
  // whether the split is the caller's or update()'s own.
  constexpr std::uint8_t lg_k  = 11;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(3'000'000, 0xfeedfaceULL);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  datasketches::cuda::theta_sketch<std::uint64_t> single(stream, mr, lg_k, seed);
  single.update(stream, device_keys.begin(), device_keys.end());

  datasketches::cuda::theta_sketch<std::uint64_t> split(stream, mr, lg_k, seed);
  const std::size_t batch = 700'000;
  for (std::size_t offset = 0; offset < device_keys.size(); offset += batch) {
    const auto end = std::min(offset + batch, device_keys.size());
    split.update(stream, device_keys.begin() + offset, device_keys.begin() + end);
  }

  REQUIRE(single.serialize_compact(stream) == split.serialize_compact(stream));
  REQUIRE(single.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta chunk sizing follows the survivor budget", "[theta][chunking][budget]")
{
  // Host-only arithmetic of the chunk policy, checked on the implementation
  // directly so the test can set theta without feeding keys.
  using impl_type            = datasketches::cuda::detail::theta::sketch_impl<std::uint64_t>;
  constexpr auto max_theta   = datasketches::cuda::detail::theta::max_theta;
  constexpr std::size_t huge = std::size_t{1} << 40;

  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  SECTION("small k gets the 2^22 floor, large k gets 16 k")
  {
    impl_type small(stream, mr, 12, 9001, 1.0F);
    REQUIRE(small.survivor_budget_() == (std::size_t{1} << 22));
    impl_type large(stream, mr, 20, 9001, 1.0F);
    REQUIRE(large.survivor_budget_() == (std::size_t{16} << 20));
  }

  // Largest chunk whose expected survivors, at the given survivor fraction,
  // fit the budget with the margin to spare.
  const auto fits = [](std::size_t budget, double rate) {
    return static_cast<std::size_t>(std::floor(budget / (impl_type::budget_margin * rate)));
  };

  SECTION("chunks at maximum theta double up to the budget less its margin")
  {
    impl_type impl(stream, mr, 12, 9001, 1.0F);
    const auto budget = impl.survivor_budget_();
    REQUIRE(impl.next_chunk_(huge, 0, -1.0) == impl_type::min_chunk_keys);
    REQUIRE(impl.next_chunk_(huge, impl_type::min_chunk_keys, -1.0) ==
            2 * impl_type::min_chunk_keys);
    REQUIRE(impl.next_chunk_(huge, budget, -1.0) == fits(budget, 1.0));
    REQUIRE(fits(budget, 1.0) < budget);
    REQUIRE(impl.next_chunk_(budget / 4, budget, -1.0) == budget / 4);
  }

  SECTION("a low duplicate factor lets chunks grow past the budget, doubling first")
  {
    impl_type impl(stream, mr, 12, 9001, 1.0F);
    const auto budget = impl.survivor_budget_();
    // One key in 1024 survives the duplicate filter. Hedged twofold that is an
    // expected 1/512 of the chunk, but at maximum theta chunks still only double.
    REQUIRE(impl.next_chunk_(huge, budget, 1.0 / 1024) == 2 * budget);
    REQUIRE(impl.next_chunk_(huge, 256 * budget, 1.0 / 1024) == fits(budget, 1.0 / 512));
    REQUIRE(impl.next_chunk_(huge, 512 * budget, 1.0 / 1024) == fits(budget, 1.0 / 512));
    // A duplicate factor at or above one half is hedged to one: it cannot
    // loosen the estimate beyond what theta alone allows.
    REQUIRE(impl.next_chunk_(huge, budget, 0.5) == fits(budget, 1.0));
    REQUIRE(impl.next_chunk_(huge, budget, 4.0) == fits(budget, 1.0));
  }

  SECTION("chunks below maximum theta are sized from theta and the duplicate factor")
  {
    impl_type impl(stream, mr, 12, 9001, 1.0F);
    const auto budget = impl.survivor_budget_();
    impl.theta_       = max_theta / 8;  // pass rate 1/8 for distinct keys
    REQUIRE(impl.next_chunk_(huge, 0, -1.0) == fits(budget, 1.0 / 8));
    REQUIRE(impl.next_chunk_(3 * budget, 0, -1.0) == 3 * budget);
    // A duplicate factor of 1/64, hedged to 1/32, scales the expectation to 1/256.
    REQUIRE(impl.next_chunk_(huge, 0, 1.0 / 64) == fits(budget, 1.0 / 256));
    // Leaning on a small duplicate factor caps growth over the previous chunk.
    REQUIRE(impl.next_chunk_(huge, budget, 1.0 / 64) == impl_type::max_chunk_growth * budget);
    REQUIRE(impl.next_chunk_(huge, 8 * budget, 1.0 / 64) == fits(budget, 1.0 / 256));
    // With a duplicate factor of one half or more no overflow is possible, so
    // growth is not capped.
    REQUIRE(impl.next_chunk_(huge, impl_type::min_chunk_keys, 0.75) == fits(budget, 1.0 / 8));
    REQUIRE(impl.next_chunk_(huge, impl_type::min_chunk_keys, -1.0) == fits(budget, 1.0 / 8));
    // A pass with no survivors predicts none: take the whole remainder.
    REQUIRE(impl.next_chunk_(huge, 0, 0.0) == huge);
  }

  SECTION("a tiny theta returns the whole remainder without overflowing size_t")
  {
    impl_type impl(stream, mr, 12, 9001, 1.0F);
    impl.theta_ = max_theta >> 50;
    REQUIRE(impl.next_chunk_(huge, 0, -1.0) == huge);
  }

  SECTION("a shrunken budget still enforces the minimum chunk below maximum theta")
  {
    impl_type impl(stream, mr, 12, 9001, 1.0F);
    impl.set_survivor_budget_(64);
    REQUIRE(impl.next_chunk_(huge, 0, -1.0) == fits(64, 1.0));
    REQUIRE(impl.next_chunk_(huge, 58, -1.0) == fits(64, 1.0));
    impl.set_survivor_budget_(1);
    REQUIRE(impl.next_chunk_(huge, 0, -1.0) == 1);
    impl.set_survivor_budget_(64);
    impl.theta_ = max_theta / 2;
    REQUIRE(impl.next_chunk_(huge, 0, -1.0) == impl_type::min_chunk_keys);
    REQUIRE_THROWS_AS(impl.set_survivor_budget_(0), std::invalid_argument);
  }
}

TEST_CASE("Theta grouped duplicates followed by distinct keys match CPU",
          "[theta][parity][chunking][budget]")
{
  // The first part of the batch repeats each key 2048 times in a row, so the
  // screen's duplicate filter removes nearly all of it and the measured
  // duplicate factor lets chunks grow far past the survivor budget. The second
  // part is all distinct keys and arrives while theta is still at its maximum,
  // so the chunk that straddles the boundary produces more survivors than the
  // budget holds: the screen must overflow, install nothing, and the retry
  // must land the same result as the CPU sketch with the default budget.
  using impl_type              = datasketches::cuda::detail::theta::sketch_impl<std::uint64_t>;
  constexpr std::uint8_t lg_k  = 12;
  constexpr std::uint64_t seed = 9001;
  constexpr std::size_t run    = 2048;
  constexpr std::size_t grouped_distinct = 4096;
  constexpr std::size_t grouped_keys     = run * grouped_distinct;  // 8M keys
  constexpr std::size_t distinct_keys    = std::size_t{1} << 24;    // 16M keys
  std::vector<std::uint64_t> keys(grouped_keys + distinct_keys);
  for (std::size_t i = 0; i < grouped_keys; ++i)
    keys[i] = 0x1000'0000'0000'0000ULL + i / run;
  for (std::size_t i = 0; i < distinct_keys; ++i)
    keys[grouped_keys + i] = 0x2000'0000'0000'0000ULL + i;

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  impl_type gpu(stream, mr, lg_k, seed, 1.0F);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.screen_overflows_ >= 1);
  REQUIRE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == (std::size_t{1} << lg_k));
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta screen overflow retries preserve the result", "[theta][parity][chunking][budget]")
{
  // A budget of 64 hashes is far below the minimum chunk, so once theta drops
  // below its maximum every chunk overflows the screen buffer and has to be
  // halved until it fits. The result must not depend on any of that: it must
  // match both the CPU sketch and a GPU sketch with the default budget.
  using impl_type              = datasketches::cuda::detail::theta::sketch_impl<std::uint64_t>;
  constexpr std::uint8_t lg_k  = 8;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(20'000, 0x0ff10adULL);
  keys.insert(keys.end(), keys.begin(), keys.begin() + 5'000);  // duplicates
  std::shuffle(keys.begin(), keys.end(), std::mt19937_64{0x5eedULL});

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);

  impl_type tiny(stream, mr, lg_k, seed, 1.0F);
  tiny.set_survivor_budget_(64);
  tiny.update(stream, device_keys.begin(), device_keys.end());
  REQUIRE(tiny.screen_overflows_ > 0);
  REQUIRE(tiny.is_estimation_mode());
  REQUIRE(tiny.get_num_retained() == (std::size_t{1} << lg_k));

  impl_type regular(stream, mr, lg_k, seed, 1.0F);
  regular.update(stream, device_keys.begin(), device_keys.end());
  REQUIRE(regular.screen_overflows_ == 0);

  const auto bytes = tiny.serialize_compact(stream);
  REQUIRE(bytes == regular.serialize_compact(stream));
  REQUIRE(bytes == theta_test::cpu_image(keys, lg_k, seed));

  // The same holds when the keys arrive in several calls, so the retry path
  // composes with the caller's own batching.
  impl_type batched(stream, mr, lg_k, seed, 1.0F);
  batched.set_survivor_budget_(64);
  const std::size_t batch = 7'001;
  for (std::size_t offset = 0; offset < device_keys.size(); offset += batch) {
    const auto end = std::min(offset + batch, device_keys.size());
    batched.update(stream, device_keys.begin() + offset, device_keys.begin() + end);
  }
  REQUIRE(batched.serialize_compact(stream) == bytes);
}

TEST_CASE("Theta batch far above the budget with few distinct keys matches CPU",
          "[theta][parity][chunking][budget]")
{
  // Fewer than k distinct keys keeps theta at its maximum for the whole batch.
  // The keys cycle with a period longer than a screen tile, so the duplicate
  // filter catches few of them and every chunk stays within the survivor
  // budget. The batch is three times that budget and must never overflow.
  using impl_type                = datasketches::cuda::detail::theta::sketch_impl<std::uint64_t>;
  constexpr std::uint8_t lg_k    = 12;
  constexpr std::uint64_t seed   = 9001;
  constexpr std::size_t count    = std::size_t{3} << 22;
  constexpr std::size_t distinct = 1'000;
  const auto pool                = make_keys(distinct, 0xd15c0ULL);
  std::vector<std::uint64_t> keys(count);
  for (std::size_t i = 0; i < count; ++i)
    keys[i] = pool[(i * 7919) % distinct];

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  impl_type gpu(stream, mr, lg_k, seed, 1.0F);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.screen_overflows_ == 0);
  REQUIRE_FALSE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == distinct);
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
}

TEST_CASE("Theta large lg_k budget of 16 k matches CPU", "[theta][parity][chunking][budget]")
{
  // At lg_k = 20 the budget is 16 k = 2^24 hashes, above the 2^22 floor, so a
  // batch of a few million keys is screened in one chunk at maximum theta and
  // trimmed to k once.
  using impl_type              = datasketches::cuda::detail::theta::sketch_impl<std::uint64_t>;
  constexpr std::uint8_t lg_k  = 20;
  constexpr std::uint64_t seed = 9001;
  auto keys                    = make_keys(5'000'000, 0xb16b00b5ULL);

  thrust::device_vector<std::uint64_t> device_keys = keys;
  ::cuda::stream stream{::cuda::devices[0]};
  auto mr = ::cuda::device_default_memory_pool(::cuda::devices[0]);
  impl_type gpu(stream, mr, lg_k, seed, 1.0F);
  gpu.update(stream, device_keys.begin(), device_keys.end());

  REQUIRE(gpu.screen_overflows_ == 0);
  REQUIRE(gpu.is_estimation_mode());
  REQUIRE(gpu.get_num_retained() == (std::size_t{1} << lg_k));
  REQUIRE(gpu.serialize_compact(stream) == theta_test::cpu_image(keys, lg_k, seed));
}

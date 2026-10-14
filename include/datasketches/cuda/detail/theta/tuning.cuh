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

namespace datasketches::cuda::detail::theta {

struct tuning {
  static constexpr int block_threads        = 256;
  static constexpr int items_per_thread     = 8;
  static constexpr std::size_t filter_slots = 1024;
  static constexpr std::size_t tile_keys    = block_threads * items_per_thread;

  // Amortize launches while bounding sort and screen scratch independently of input size.
  static constexpr std::size_t chunk_target_multiple = 16;
  static constexpr std::size_t min_chunk_keys        = std::size_t{1} << 20;
  static constexpr std::size_t min_survivor_budget   = std::size_t{1} << 22;
  static constexpr std::size_t max_chunk_growth      = 64;
  static constexpr double budget_margin              = 1.1;
  static constexpr double duplicate_hedge            = 2.0;

  static unsigned int grid_size(std::size_t count) noexcept
  {
    constexpr auto max_blocks = std::size_t{0x7fffffff};
    const auto blocks         = (count + tile_keys - 1) / tile_keys;
    return static_cast<unsigned int>(blocks < max_blocks ? blocks : max_blocks);
  }
};

}  // namespace datasketches::cuda::detail::theta

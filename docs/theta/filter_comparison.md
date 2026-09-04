<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Theta block-local duplicate filter comparison

The Theta screen kernel can deduplicate hashes inside a block before they reach
the radix sort. This records what each candidate structure is worth, measured
rather than argued, and is the evidence behind choosing one.

## Measurement setup

- NVIDIA B200, driver 580.82.07, CUDA 13.0.48, GCC 13.3.0, CMake 4.1.2
- CCCL `cba1df57`, nvbench pinned to `410dcdd2`
- 120 configurations per variant: `{cold, warm} x {U32, U64, F32, F64} x
  {2^20, 2^24, 10^8} keys x 6 arrival distributions x lg_k {12, 20}`
- Speedup is `median_gpu_time(no filter) / median_gpu_time(candidate)`, so above
  1 is faster
- One variant at a time on one idle GPU. Running several concurrently on
  separate idle GPUs of the same host shifted every number by about 27% without
  changing the ordering, so those runs were discarded.

The six distributions hold the duplicate count fixed at 1% distinct (except
`unique`) and vary only where the duplicates sit: `scatter` spaces repeats one
distinct-count apart, `shuffled` randomizes them, `warp_run` and `block_run`
group them into runs of 32 and 2048, and `zipf` gives them power-law
frequencies. Separating duplicate count from duplicate locality is the point. A
benchmark that only generates scattered repeats cannot tell an optimization that
exploits locality from one that does nothing.

## Headline: three interleaved repeats

Run-to-run spread on identical code is about 0.7%, so differences below roughly
1.5% are not meaningful.

| Filter | mean geomean | worst | local dups | no local dups | warm only |
|---|---|---|---|---|---|
| direct-mapped, 1024 slots | **1.194** | 0.79 | 1.549 | 0.921 | 1.083 |
| cuco set, retrieve_all, 2048 slots | **1.168** | 0.76 | 1.530 | 0.892 | 1.059 |
| cuco set, guarded, 1024 slots | 1.145 | 0.70 | 1.497 | 0.877 | 1.039 |
| cuco map, guarded, 1024 slots | 1.073 | 0.55 | 1.404 | 0.820 | 0.968 |

Three successive redesigns of the cuco filter, each measured: map to key-only set
(1.073 to 1.145), guard to a guaranteed 2:1 capacity ratio (1.145 to 1.162), and
emit-as-you-go to bulk retrieval (1.162 to 1.168).

### Open-addressing set against the current direct-mapped cache

| | |
|---|---|
| geomean, set relative to direct-mapped | **0.978** |
| direct-mapped faster by | **2.3%** |
| configurations the set wins | 31 of 240 |
| set best case | 1.14x |
| set worst case | 0.87x |

## At scale the gap widens

The 120-configuration matrix tops out at 100M keys, where the ~130 us of
per-update fixed overhead is still 14% of the call and neither path is
asymptotic. Sweeping to 2B keys (16 GB of input), U64, cold, `lg_k = 12`,
throughput in Gkeys/s:

| keys | dist | direct-mapped | cuco set | cuco vs direct |
|---|---|---|---|---|
| 100M | unique | 107 | 103 | -4% |
| 2B | unique | 217 | 199 | **-8%** |
| 100M | zipf | 100 | 92 | -9% |
| 2B | zipf | 194 | 169 | **-13%** |
| 100M | block_run | 79 | 82 | +4% |
| 2B | block_run | 221 | 212 | -4% |

Both scale the same way and neither falls off a cliff, but the cuco set's extra
per-key cost, 140 instructions against 81 and a retrieval proportional to
capacity, is a fixed tax that the fixed overhead was masking at 100M. Once that
overhead amortizes, the tax is what is left. The 2.3% geomean deficit reported
below is a 100M-key number; at production scale the honest figure is 8 to 13
percent on realistic distributions, and the one case the set wins at 100M,
`block_run`, it loses at 2B.

## Three things that made the cuco filter faster

**Set instead of map, worth 6.7%.** `fixed_capacity_map_ref` stores a
`pair<uint64_t, uint32_t>` slot that pads to 16 bytes, and this filter never
reads the payload. A key-only 8-byte slot halves the table and removes one atomic
per insert, since the map does a 64-bit CAS for the key and a 32-bit CAS for the
payload. This required reaching past the public API.

**Guaranteed capacity instead of a policed one, worth 1.6%.** Halving the tile to
1024 keys against a 2048-slot table makes the load at most one half by
construction, so the occupancy counter, its per-key load and its per-insert
atomic all disappear. The correctness argument becomes checkable by inspection.

**Bulk retrieval instead of per-key emission, worth 0.5%.** Insertion becomes the
output and the tile's occupied slots are retrieved at the end. This removes the
per-key register staging, which dropped registers from 40 to 31.

### What did not work

**`__launch_bounds__` to force fewer registers: 0.800**, a catastrophic
regression from spilling.

**Vectorized clear and retrieval, plus an empty-table fast path: 1.147 against
1.168.** Two lessons. Pairing slots into 128-bit accesses halves *instructions*
but moves identical *traffic*, and shared wavefronts were the constraint, so it
bought nothing. And the empty-table fast path almost never fires: once theta has
tightened there are still a few survivors per block, not zero, so blocks pay the
full scan anyway plus the new bookkeeping.

The retrieval cost is therefore structural. Skipping it requires knowing which
slots are occupied, which requires an insert that returns where it landed.

## Broader field, single run each

| Variant | geomean | worst | local dups | no local dups | warm only | correct |
|---|---|---|---|---|---|---|
| warp collapse only | 1.211 | 0.80 | 1.523 | 0.963 | 1.136 | yes |
| direct-mapped 1024 | 1.191 | 0.79 | 1.545 | 0.917 | 1.078 | yes |
| direct-mapped 2048 | 1.185 | 0.78 | 1.538 | 0.913 | 1.072 | yes |
| direct-mapped 4096 | 1.168 | 0.78 | 1.513 | 0.901 | 1.056 | yes |
| 2-way set associative | 1.138 | 0.74 | 1.456 | 0.890 | 1.026 | yes |
| 4-way set associative | 1.121 | 0.72 | 1.440 | 0.874 | 1.014 | yes |
| cuco map, bucket 1 | 1.096 | 0.57 | 1.440 | 0.835 | 0.997 | yes |
| cuco map, load 25% | 1.077 | 0.55 | 1.405 | 0.825 | 0.970 | yes |
| cuco map, bucket 4 | 1.055 | 0.49 | 1.384 | 0.804 | 0.954 | yes |
| cuco map, bucket 8 | 1.036 | 0.40 | 1.372 | 0.781 | 0.940 | yes |
| cuco map, load 75% | 1.026 | 0.53 | 1.373 | 0.767 | 0.940 | yes |
| cuco with `__launch_bounds__` | 0.800 | | 1.056 | 0.607 | 0.659 | yes |
| cuco, unguarded | 0.649 | **0.08** | 1.178 | 0.358 | 0.784 | **no** |

## Screen kernel counters

First screen launch, 100M keys, `Dist=shuffled`, `lg_k=20`, identical across
variants at 8192 blocks of 256 threads.

| Variant | registers | occupancy limit, blocks (reg / smem / warp) | achieved | instructions | shared ld | shared atomic |
|---|---|---|---|---|---|---|
| no filter | 32 | 8 / unbounded / 8 | 87.4% | 45.0 M | 0 | 0 |
| warp only | 32 | 8 / unbounded / 8 | 92.3% | 51.5 M | 0 | 0 |
| direct-mapped | 40 | 6 / 14 / 8 | 75% | 68.4 M | 3.24 M | 0 |
| 4-way associative | 42 | 6 / 14 / 8 | 57.8% | 93.7 M | 25.0 M | 0 |
| cuco map | 44 | **5** / 7 / 8 | 62.5% | 166 M | 4.97 M | 13.6 M |
| cuco unguarded | 44 | 5 / 7 / 8 | 45.1% | **7 122 M** | **3 318 M** | 13.6 M |

Registers, not shared memory, are the binding occupancy limit. Probing is not the
bottleneck for the guarded table: its shared load count is within 1.5x of the
direct-mapped cache.

## Why the guard exists

`insert` returns a bare `bool`, and a false result means either "already present"
or "no room left". Those require opposite actions. Treating full as duplicate
drops distinct hashes, which is a silent wrong answer, not a slowdown; the
unguarded variant fails the Theta parity test. Emitting on every false gives up
nearly all deduplication.

The guard prevents the ambiguity instead of resolving it. A block-scope counter
tracks occupied slots, and above half load the filter stops consulting the table
and emits. No probe ever runs against a table that could be full, so a failed
insert is unambiguously a duplicate. The counter is read before the insert and
incremented after, so it can only lag by the inserts in flight, bounded by the
block width, keeping true load under 75% in the worst interleaving.

Sizing the table to avoid this instead is not available. A block is handed 2048
keys, so holding them all at 50% load needs 4096 slots; at the map's 16-byte slot
that is 64 KiB against a 48 KiB per-block limit. The choices are to let it fill,
to reset it and pay a barrier each time, or to stop using it when full.

## What cuco would need for this to be a clean win

1. **A public key-only set ref over caller-provided storage.** The only item here
   with a measured number behind it: 8-byte slots are worth 6.7%.
2. **An insert that separates duplicate from table-full.** The legacy
   cuCollections `insert_and_find` already does, returning `end()` on a full
   table; the cudax map ref does not. This removes the need for an occupancy
   guard to establish correctness.
3. **A bounded-probe insert**, something like
   `try_insert(key, max_probes) -> {inserted, duplicate, gave_up}`. Neither
   library has this. It is the most valuable of the three for a best-effort
   filter, because a probe budget caps worst-case cost structurally rather than
   by external bookkeeping, and would make the guard unnecessary entirely.
4. A cooperative `initialize` / `clear` on the ref, which legacy cuco has and
   cudax does not, for the per-tile reset this filter performs by hand.
5. **An insert that returns the slot it claimed.** Retrieval currently costs
   O(capacity) per block no matter how few hashes it finds, and measurement
   showed that cost cannot be optimized away without knowing which slots are
   live. Legacy `insert_and_find` returns an iterator, which is exactly that.
6. `retrieve_all` itself, which legacy cuco offers for device-wide containers and
   cudax does not offer at all.

Not needed: erase, `find`, a host-side owning container, cooperative group size
above 1, dynamic capacity, or multimap semantics.

## Conclusion

The exact set does what it was designed to do. It never falls off the
open-addressing cliff, it is correct on every configuration, and against the
duplicate-heavy inputs it exists to serve it reaches 1.530, within 1.2% of the
direct-mapped cache. Three redesigns closed the overall gap from 5.6% to 2.3% at 100M keys; at 2B
keys the gap is 8 to 13 percent on realistic inputs, see "At scale the gap widens".

It is still behind a direct-mapped cache that costs ten lines and no library
dependency, for a reason no amount of tuning addresses: the duplicates worth
catching were mostly already removed by the free `__match_any_sync` stage that
runs before either structure, so an exact set is paying for precision that has
little left to find.

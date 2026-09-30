// SPDX-License-Identifier: MIT

//! Four independent SipHash-2-4-128 OutPoint fingerprints using ARM NEON.
//!
//! Each vector holds two hashes. Two vector states interleave their round operations
//! to expose independent work without changing the standard hashes or their keys.
//! Cached scalar txid prefixes can be packed into the same kernel for output hashing.
#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

use crate::sip128::TxidPrefix;

/// Two NEON vectors per state word, with hash lanes `[0, 1]` and `[2, 3]`.
struct State {
    v0: [uint64x2_t; 2],
    v1: [uint64x2_t; 2],
    v2: [uint64x2_t; 2],
    v3: [uint64x2_t; 2],
}

#[inline(always)]
fn rotate<const LEFT: i32, const RIGHT: i32>(value: uint64x2_t) -> uint64x2_t {
    // SAFETY: NEON is baseline on AArch64, and each call uses complementary shifts.
    unsafe { vsriq_n_u64::<RIGHT>(vshlq_n_u64::<LEFT>(value), value) }
}

#[inline(always)]
fn rotate32(value: uint64x2_t) -> uint64x2_t {
    // SAFETY: This swaps the 32-bit halves within each independent 64-bit hash lane.
    unsafe { vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(value))) }
}

#[inline(always)]
fn pack(words: [u64; 4]) -> [uint64x2_t; 2] {
    // SAFETY: Each baseline NEON load reads two of the four valid array elements.
    unsafe { [vld1q_u64(words.as_ptr()), vld1q_u64(words.as_ptr().add(2))] }
}

impl State {
    #[inline(always)]
    fn new([k0, k1]: [u64; 2]) -> Self {
        // SAFETY: These baseline NEON instructions initialize all four hash lanes.
        unsafe {
            Self {
                v0: [vdupq_n_u64(k0 ^ 0x736f6d6570736575); 2],
                v1: [vdupq_n_u64(k1 ^ 0x646f72616e646f6d ^ 0xee); 2],
                v2: [vdupq_n_u64(k0 ^ 0x6c7967656e657261); 2],
                v3: [vdupq_n_u64(k1 ^ 0x7465646279746573); 2],
            }
        }
    }

    #[inline(always)]
    fn from_prefixes(prefixes: [TxidPrefix; 4]) -> Self {
        let words = prefixes.map(TxidPrefix::state_words);
        Self {
            v0: pack(words.map(|state| state[0])),
            v1: pack(words.map(|state| state[1])),
            v2: pack(words.map(|state| state[2])),
            v3: pack(words.map(|state| state[3])),
        }
    }

    #[inline(always)]
    fn round(&mut self) {
        // SAFETY: All instructions are baseline NEON. Each operation runs on both
        // independent vector states before the next dependency, with no lane mixing.
        unsafe {
            self.v0[0] = vaddq_u64(self.v0[0], self.v1[0]);
            self.v0[1] = vaddq_u64(self.v0[1], self.v1[1]);
            self.v1[0] = veorq_u64(rotate::<13, 51>(self.v1[0]), self.v0[0]);
            self.v1[1] = veorq_u64(rotate::<13, 51>(self.v1[1]), self.v0[1]);
            self.v0[0] = rotate32(self.v0[0]);
            self.v0[1] = rotate32(self.v0[1]);
            self.v2[0] = vaddq_u64(self.v2[0], self.v3[0]);
            self.v2[1] = vaddq_u64(self.v2[1], self.v3[1]);
            self.v3[0] = veorq_u64(rotate::<16, 48>(self.v3[0]), self.v2[0]);
            self.v3[1] = veorq_u64(rotate::<16, 48>(self.v3[1]), self.v2[1]);
            self.v0[0] = vaddq_u64(self.v0[0], self.v3[0]);
            self.v0[1] = vaddq_u64(self.v0[1], self.v3[1]);
            self.v3[0] = veorq_u64(rotate::<21, 43>(self.v3[0]), self.v0[0]);
            self.v3[1] = veorq_u64(rotate::<21, 43>(self.v3[1]), self.v0[1]);
            self.v2[0] = vaddq_u64(self.v2[0], self.v1[0]);
            self.v2[1] = vaddq_u64(self.v2[1], self.v1[1]);
            self.v1[0] = veorq_u64(rotate::<17, 47>(self.v1[0]), self.v2[0]);
            self.v1[1] = veorq_u64(rotate::<17, 47>(self.v1[1]), self.v2[1]);
            self.v2[0] = rotate32(self.v2[0]);
            self.v2[1] = rotate32(self.v2[1]);
        }
    }

    #[inline(always)]
    fn words(&mut self, words: [u64; 4]) {
        let [first, second] = pack(words);
        // SAFETY: Baseline NEON handles one independent message word per hash lane.
        unsafe {
            self.v3[0] = veorq_u64(self.v3[0], first);
            self.v3[1] = veorq_u64(self.v3[1], second);
            self.round();
            self.round();
            self.v0[0] = veorq_u64(self.v0[0], first);
            self.v0[1] = veorq_u64(self.v0[1], second);
        }
    }

    #[inline(always)]
    fn output(&self) -> [u64; 4] {
        // SAFETY: Baseline NEON computes each hash separately, and each store writes
        // two elements inside the valid four-element output array.
        unsafe {
            let first = veorq_u64(
                veorq_u64(self.v0[0], self.v1[0]),
                veorq_u64(self.v2[0], self.v3[0]),
            );
            let second = veorq_u64(
                veorq_u64(self.v0[1], self.v1[1]),
                veorq_u64(self.v2[1], self.v3[1]),
            );
            let mut result = [0; 4];
            vst1q_u64(result.as_mut_ptr(), first);
            vst1q_u64(result.as_mut_ptr().add(2), second);
            result
        }
    }

    #[inline(always)]
    fn finish(mut self, vouts: [u32; 4]) -> [(u64, u64); 4] {
        self.words(vouts.map(|vout| (36u64 << 56) | u64::from(vout)));
        // SAFETY: The standard SipHash128 finalization constants are applied to
        // every independent hash using baseline NEON, without changing lane order.
        unsafe {
            self.v2[0] = veorq_u64(self.v2[0], vdupq_n_u64(0xee));
            self.v2[1] = veorq_u64(self.v2[1], vdupq_n_u64(0xee));
            for _ in 0..4 {
                self.round();
            }
            let low = self.output();
            self.v1[0] = veorq_u64(self.v1[0], vdupq_n_u64(0xdd));
            self.v1[1] = veorq_u64(self.v1[1], vdupq_n_u64(0xdd));
            for _ in 0..4 {
                self.round();
            }
            let high = self.output();
            core::array::from_fn(|i| (low[i], high[i]))
        }
    }
}

/// Hashes four complete OutPoints using two interleaved two-lane NEON states.
#[inline(always)]
pub fn hash(keys: [u64; 2], txids: [&[u8; 32]; 4], vouts: [u32; 4]) -> [(u64, u64); 4] {
    let mut state = State::new(keys);
    for word in 0..4 {
        let start = word * 8;
        state.words(
            txids.map(|txid| {
                u64::from_le_bytes(txid[start..start + 8].try_into().expect("txid word"))
            }),
        );
    }
    state.finish(vouts)
}

/// Finishes four cached txid prefixes, possibly belonging to different transactions.
#[inline(always)]
pub fn finish(prefixes: [TxidPrefix; 4], vouts: [u32; 4]) -> [(u64, u64); 4] {
    State::from_prefixes(prefixes).finish(vouts)
}

#[cfg(test)]
mod tests {
    use siphasher::sip128::SipHasher24;

    use super::*;

    fn reference(keys: [u64; 2], txid: &[u8; 32], vout: u32) -> (u64, u64) {
        let mut message = [0; 36];
        message[..32].copy_from_slice(txid);
        message[32..].copy_from_slice(&vout.to_le_bytes());
        let result = SipHasher24::new_with_keys(keys[0], keys[1]).hash(&message);
        (result.h1, result.h2)
    }

    fn check(keys: [u64; 2], txids: [&[u8; 32]; 4], vouts: [u32; 4]) {
        let expected = core::array::from_fn(|i| reference(keys, txids[i], vouts[i]));
        let prefixes = txids.map(|txid| TxidPrefix::new(keys, txid));
        assert_eq!(hash(keys, txids, vouts), expected);
        assert_eq!(finish(prefixes, vouts), expected);
        assert_eq!(
            finish(prefixes, vouts.map(|vout| !vout)),
            core::array::from_fn(|i| reference(keys, txids[i], !vouts[i])),
        );
        for permutation in [[1, 0, 3, 2], [2, 3, 0, 1], [3, 1, 0, 2]] {
            let swapped_vouts = permutation.map(|i| vouts[i]);
            let swapped_expected = permutation.map(|i| expected[i]);
            assert_eq!(
                hash(keys, permutation.map(|i| txids[i]), swapped_vouts),
                swapped_expected,
            );
            assert_eq!(
                finish(permutation.map(|i| prefixes[i]), swapped_vouts),
                swapped_expected,
            );
        }
    }

    #[test]
    fn edge_keys_vouts_and_shared_txids_match_reference() {
        let txids = [[0; 32], [0xff; 32], [0x55; 32], [0xaa; 32]];
        for keys in [[0; 2], [u64::MAX; 2], [0, u64::MAX], [1, 2]] {
            for vout in [0, 1, 255, 256, 0x12345678, u32::MAX] {
                check(keys, txids.each_ref(), [vout, !vout, vout ^ 0x55, 0]);
                for txid in &txids {
                    check(keys, [txid; 4], [vout, !vout, 0, u32::MAX]);
                }
            }
        }
    }

    #[test]
    fn varied_distinct_outpoints_and_reused_prefixes_match_reference() {
        let mut seed = 0xb6e29a81375c04dfu64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..512 {
            let keys = [next(), next()];
            let mut txids = [[0; 32]; 4];
            for txid in &mut txids {
                for word in txid.chunks_exact_mut(8) {
                    word.copy_from_slice(&next().to_le_bytes());
                }
            }
            let vouts = core::array::from_fn(|_| {
                u32::from_le_bytes(next().to_le_bytes()[..4].try_into().unwrap())
            });
            check(keys, txids.each_ref(), vouts);
            check(keys, [&txids[0]; 4], vouts);
        }
    }
}

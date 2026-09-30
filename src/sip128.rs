// SPDX-License-Identifier: MIT

//! SipHash-2-4-128 for the fixed 36-byte OutPoint encoding `txid || vout_le`.
//! Outputs sharing a txid reuse its compressed state, then finalize with the vout.

#[derive(Clone, Copy)]
struct State {
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
}

impl State {
    #[inline]
    fn new([k0, k1]: [u64; 2]) -> Self {
        Self {
            v0: k0 ^ 0x736f6d6570736575,
            v1: k1 ^ 0x646f72616e646f6d ^ 0xee,
            v2: k0 ^ 0x6c7967656e657261,
            v3: k1 ^ 0x7465646279746573,
        }
    }

    #[inline]
    fn round(&mut self) {
        self.v0 = self.v0.wrapping_add(self.v1);
        self.v1 = self.v1.rotate_left(13) ^ self.v0;
        self.v0 = self.v0.rotate_left(32);
        self.v2 = self.v2.wrapping_add(self.v3);
        self.v3 = self.v3.rotate_left(16) ^ self.v2;
        self.v0 = self.v0.wrapping_add(self.v3);
        self.v3 = self.v3.rotate_left(21) ^ self.v0;
        self.v2 = self.v2.wrapping_add(self.v1);
        self.v1 = self.v1.rotate_left(17) ^ self.v2;
        self.v2 = self.v2.rotate_left(32);
    }

    #[inline]
    fn word(&mut self, word: u64) {
        self.v3 ^= word;
        self.round();
        self.round();
        self.v0 ^= word;
    }

    #[inline]
    fn output(&self) -> u64 {
        self.v0 ^ self.v1 ^ self.v2 ^ self.v3
    }
}

/// Hash state after the 32-byte txid, reusable for its output indices.
#[derive(Clone, Copy)]
pub struct TxidPrefix(State);

impl TxidPrefix {
    /// Exposes the cached state for packing four prefixes into the ARM SIMD kernel.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub(crate) fn state_words(self) -> [u64; 4] {
        [self.0.v0, self.0.v1, self.0.v2, self.0.v3]
    }

    /// Compresses the txid's four little-endian words using the session key.
    #[inline]
    pub fn new(keys: [u64; 2], txid: &[u8; 32]) -> Self {
        let mut state = State::new(keys);
        for word in txid.chunks_exact(8) {
            state.word(u64::from_le_bytes(word.try_into().expect("txid word")));
        }
        Self(state)
    }

    /// Appends the vout and message length, then returns the two 64-bit tag halves.
    #[inline]
    pub fn finish(self, vout: u32) -> (u64, u64) {
        let mut state = self.0;
        state.word((36u64 << 56) | u64::from(vout));
        state.v2 ^= 0xee;
        for _ in 0..4 {
            state.round();
        }
        let low = state.output();
        state.v1 ^= 0xdd;
        for _ in 0..4 {
            state.round();
        }
        (low, state.output())
    }
}

/// Hashes one complete OutPoint without reusing a txid prefix.
#[inline]
pub fn hash(keys: [u64; 2], txid: &[u8; 32], vout: u32) -> (u64, u64) {
    TxidPrefix::new(keys, txid).finish(vout)
}

#[cfg(test)]
mod tests {
    use siphasher::sip128::SipHasher24;

    use super::*;

    #[test]
    fn matches_independent_siphasher() {
        for keys in [[0, 0], [1, 2], [u64::MAX, u64::MAX]] {
            for txid in [
                [0; 32],
                [0xff; 32],
                core::array::from_fn(|i| u8::try_from(i).unwrap()),
            ] {
                let prefix = TxidPrefix::new(keys, &txid);
                for vout in [0, 1, 255, 256, 0x12345678, u32::MAX] {
                    let mut message = [0; 36];
                    message[..32].copy_from_slice(&txid);
                    message[32..].copy_from_slice(&vout.to_le_bytes());
                    let expected = SipHasher24::new_with_keys(keys[0], keys[1]).hash(&message);
                    assert_eq!(hash(keys, &txid, vout), (expected.h1, expected.h2));
                    assert_eq!(prefix.finish(vout), (expected.h1, expected.h2));
                }
            }
        }
    }
}

// SPDX-License-Identifier: MIT

//! AES-128-CMAC for the fixed 36-byte message `txid || vout_le`.
//! ARM targets compiled with AES use native instructions. Other builds use
//! RustCrypto's hardware dispatch, or its software backend with `aes_backend="soft"`.
//! Caching the two txid encryptions leaves one encryption per additional output.
//! The eight-record variant uses RustCrypto to batch independent CBC chains,
//! preserving every tag and the ordering of the three stages within each record.

pub(crate) use portable::Session as BatchedSession;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    not(feature = "software")
))]
pub(crate) use arm::Session;
#[cfg(not(all(
    target_arch = "aarch64",
    target_feature = "aes",
    not(feature = "software")
)))]
pub(crate) use portable::Session;

/// Doubles a big-endian CMAC subkey without secret-dependent branches.
fn double(block: [u8; 16]) -> [u8; 16] {
    let mut result = [0; 16];
    for index in 0..15 {
        result[index] = (block[index] << 1) | (block[index + 1] >> 7);
    }
    result[15] = (block[15] << 1) ^ (0x87 & 0u8.wrapping_sub(block[0] >> 7));
    result
}

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "aes",
    not(feature = "software")
))]
mod arm {
    use core::arch::aarch64::*;

    use super::double;

    /// Expanded key and CMAC K2 combined with the fixed final-block padding.
    pub(crate) struct Session {
        keys: [uint8x16_t; 11],
        tail: uint8x16_t,
    }

    /// Two cached CBC encryptions for outputs of one transaction.
    #[derive(Clone, Copy)]
    pub(crate) struct TxidPrepared {
        tail_state: uint8x16_t,
    }

    /// Encrypts one block with the expanded AES-128 key using ARM AES instructions.
    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn encrypt(keys: &[uint8x16_t; 11], mut state: uint8x16_t) -> uint8x16_t {
        for key in &keys[..9] {
            state = vaesmcq_u8(vaeseq_u8(state, *key));
        }
        veorq_u8(vaeseq_u8(state, keys[9]), keys[10])
    }

    #[inline]
    #[target_feature(enable = "aes")]
    unsafe fn expand_key(secret: &[u8; 16]) -> ([uint8x16_t; 11], uint8x16_t) {
        // SAFETY: This module is built only for AArch64 targets with AES enabled.
        unsafe {
            let zero = vdupq_n_u8(0);
            let mut keys = [zero; 11];
            keys[0] = vld1q_u8(secret.as_ptr());
            let mut previous = *secret;
            for (round_key, rcon) in keys[1..]
                .iter_mut()
                .zip([1, 2, 4, 8, 16, 32, 64, 128, 27, 54])
            {
                let rotated =
                    u32::from_le_bytes([previous[13], previous[14], previous[15], previous[12]]);
                let substituted = vaeseq_u8(vreinterpretq_u8_u32(vdupq_n_u32(rotated)), zero);
                let mut word = vgetq_lane_u32::<0>(vreinterpretq_u32_u8(substituted)).to_le_bytes();
                word[0] ^= rcon;
                for column in previous.chunks_exact_mut(4) {
                    for (byte, part) in column.iter_mut().zip(&mut word) {
                        *byte ^= *part;
                        *part = *byte;
                    }
                }
                *round_key = vld1q_u8(previous.as_ptr());
            }
            let l = encrypt(&keys, zero);
            (keys, l)
        }
    }

    #[inline]
    fn bytes(block: uint8x16_t) -> [u8; 16] {
        let mut result = [0u8; 16];
        // SAFETY: NEON is baseline on AArch64; the destination has 16 bytes.
        unsafe { vst1q_u8(result.as_mut_ptr(), block) };
        result
    }

    impl Session {
        /// Expands the session's AES key and derives CMAC K2 once.
        pub(crate) fn new(secret: &[u8; 16]) -> Self {
            // SAFETY: This module is selected only when the target includes ARM AES.
            unsafe {
                let (keys, l) = expand_key(secret);
                let mut tail = double(double(bytes(l)));
                tail[4] ^= 0x80;
                Self {
                    keys,
                    tail: vld1q_u8(tail.as_ptr()),
                }
            }
        }

        /// Computes a complete OutPoint's CMAC tag using three AES encryptions.
        #[inline]
        pub(crate) fn hash(&self, txid: &[u8; 32], vout: u32) -> [u8; 16] {
            // SAFETY: The target includes ARM AES and both loads stay in the txid.
            unsafe {
                let first = encrypt(&self.keys, vld1q_u8(txid.as_ptr()));
                let second = encrypt(&self.keys, veorq_u8(first, vld1q_u8(txid.as_ptr().add(16))));
                let vout = vreinterpretq_u8_u32(vsetq_lane_u32::<0>(vout.to_le(), vdupq_n_u32(0)));
                bytes(encrypt(
                    &self.keys,
                    veorq_u8(veorq_u8(second, self.tail), vout),
                ))
            }
        }

        /// Caches the first two CBC encryptions for a transaction's outputs.
        #[inline]
        pub(crate) fn prepare_txid(&self, txid: &[u8; 32]) -> TxidPrepared {
            // SAFETY: The target includes ARM AES and both loads stay in the txid.
            unsafe {
                let first = encrypt(&self.keys, vld1q_u8(txid.as_ptr()));
                let second = encrypt(&self.keys, veorq_u8(first, vld1q_u8(txid.as_ptr().add(16))));
                TxidPrepared {
                    tail_state: veorq_u8(second, self.tail),
                }
            }
        }

        /// Completes a cached txid with one AES encryption for the chosen vout.
        #[inline]
        pub(crate) fn finish(&self, prepared: TxidPrepared, vout: u32) -> [u8; 16] {
            // SAFETY: The target includes ARM AES.
            unsafe {
                let vout = vreinterpretq_u8_u32(vsetq_lane_u32::<0>(vout.to_le(), vdupq_n_u32(0)));
                bytes(encrypt(&self.keys, veorq_u8(prepared.tail_state, vout)))
            }
        }
    }
}

mod portable {
    use aes::Aes128Enc;
    use aes::Block;
    use aes::cipher::BlockCipherEncrypt;
    use aes::cipher::KeyInit;

    use super::double;

    /// RustCrypto AES key and the fixed final-block padding combined with CMAC K2.
    pub(crate) struct Session {
        cipher: Aes128Enc,
        tail: u128,
    }

    /// Two cached CBC encryptions, including the final-block padding and CMAC K2.
    #[derive(Clone, Copy)]
    pub(crate) struct TxidPrepared {
        tail_state: u128,
    }

    impl Session {
        /// Expands the AES key and derives the final-block mask once per session.
        pub(crate) fn new(secret: &[u8; 16]) -> Self {
            let cipher = Aes128Enc::new(&(*secret).into());
            let mut l = Block::default();
            cipher.encrypt_block(&mut l);
            let mut tail = double(double(l.into()));
            tail[4] ^= 0x80;
            Self {
                cipher,
                tail: u128::from_le_bytes(tail),
            }
        }

        #[inline]
        fn encrypt(&self, bytes: [u8; 16]) -> [u8; 16] {
            let mut block = Block::from(bytes);
            self.cipher.encrypt_block(&mut block);
            block.into()
        }

        /// Computes one complete OutPoint's CMAC tag.
        #[inline]
        pub(crate) fn hash(&self, txid: &[u8; 32], vout: u32) -> [u8; 16] {
            self.finish(self.prepare_txid(txid), vout)
        }

        /// Caches the two CBC encryptions shared by a transaction's outputs.
        #[inline]
        pub(crate) fn prepare_txid(&self, txid: &[u8; 32]) -> TxidPrepared {
            let first = self.encrypt(txid[..16].try_into().expect("first AES block"));
            let second = u128::from_le_bytes(first)
                ^ u128::from_le_bytes(txid[16..].try_into().expect("second AES block"));
            let state = self.encrypt(second.to_le_bytes());
            TxidPrepared {
                tail_state: u128::from_le_bytes(state) ^ self.tail,
            }
        }

        /// Completes a cached txid with one encryption for the selected vout.
        #[inline]
        pub(crate) fn finish(&self, prepared: TxidPrepared, vout: u32) -> [u8; 16] {
            self.encrypt((prepared.tail_state ^ u128::from(vout)).to_le_bytes())
        }

        /// Encrypts eight independent OutPoints, advancing all lanes one CBC stage at a time.
        #[inline]
        pub(crate) fn hash_batch(&self, txids: [&[u8; 32]; 8], vouts: [u32; 8]) -> [[u8; 16]; 8] {
            let mut blocks: [Block; 8] = txids.map(|txid| {
                Block::from(<[u8; 16]>::try_from(&txid[..16]).expect("first AES block"))
            });
            self.cipher.encrypt_blocks(&mut blocks);
            for (block, txid) in blocks.iter_mut().zip(txids) {
                for (byte, message_byte) in block.iter_mut().zip(&txid[16..]) {
                    *byte ^= message_byte;
                }
            }
            self.cipher.encrypt_blocks(&mut blocks);
            let prefixes = blocks.map(|block| TxidPrepared {
                tail_state: u128::from_le_bytes(block.into()) ^ self.tail,
            });
            self.finish_batch(prefixes, vouts)
        }

        /// Completes eight cached txids with one parallel AES call.
        #[inline]
        pub(crate) fn finish_batch(
            &self,
            prepared: [TxidPrepared; 8],
            vouts: [u32; 8],
        ) -> [[u8; 16]; 8] {
            let mut blocks: [Block; 8] = core::array::from_fn(|i| {
                Block::from((prepared[i].tail_state ^ u128::from(vouts[i])).to_le_bytes())
            });
            self.cipher.encrypt_blocks(&mut blocks);
            blocks.map(Into::into)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BatchedSession;
    use super::Session;

    #[test]
    fn batches_match_scalar_tags_and_reuse_prefixes() {
        let txids: [[u8; 32]; 8] = core::array::from_fn(|lane| {
            core::array::from_fn(|byte| {
                // Shared first halves also exercise each lane's distinct second half.
                u8::try_from(if byte < 16 { byte } else { lane * 32 + byte })
                    .expect("test byte fits u8")
            })
        });
        let vouts = [0, 1, 255, 256, 0x1234_5678, u32::MAX, 2, 65_536];
        for secret in [[0; 16], [0xff; 16], [0x6d; 16]] {
            let scalar = Session::new(&secret);
            let batch = BatchedSession::new(&secret);
            let expected = core::array::from_fn(|i| scalar.hash(&txids[i], vouts[i]));
            assert_eq!(batch.hash_batch(txids.each_ref(), vouts), expected);
            let prefixes = txids.each_ref().map(|txid| batch.prepare_txid(txid));
            assert_eq!(batch.finish_batch(prefixes, vouts), expected);

            let reversed_txids = core::array::from_fn(|i| &txids[7 - i]);
            let reversed_vouts = core::array::from_fn(|i| vouts[7 - i]);
            assert_eq!(
                batch.hash_batch(reversed_txids, reversed_vouts),
                core::array::from_fn(|i| expected[7 - i])
            );
            let next_vouts = vouts.map(|vout| vout.wrapping_add(1));
            assert_eq!(
                batch.finish_batch(prefixes, next_vouts),
                core::array::from_fn(|i| scalar.hash(&txids[i], next_vouts[i]))
            );
            assert_eq!(
                batch.finish_batch([prefixes[0]; 8], vouts),
                core::array::from_fn(|i| scalar.hash(&txids[0], vouts[i]))
            );
        }
    }

    #[cfg(feature = "software")]
    #[test]
    fn software_build_does_not_use_aes_instructions() {
        assert!(
            !aes::hardware_accelerated(),
            "software benchmarks need RUSTFLAGS='--cfg aes_backend=\"soft\"'"
        );
    }

    #[test]
    fn openssl_outpoint_vectors_and_cached_prefix() {
        // Independent OpenSSL AES-128-CMAC vectors for 36-byte OutPoints.
        let session = Session::new(&[0; 16]);
        let txid = [0; 32];
        let prepared = session.prepare_txid(&txid);
        for (vout, expected) in [
            (0, "c629a5f5228c836f688bde1af7c52ebb"),
            (1, "4cf85de30c658de25926e97d644be8c0"),
            (255, "ea19a16e96ac8df930fe8ed899526acd"),
            (256, "02cbbd9f5a601e8bfbbfc03f23e0e888"),
            (u32::MAX, "867b9371c6eb1d29c5b7356e37bae042"),
        ] {
            let expected = hex::decode(expected).expect("valid test vector");
            assert_eq!(session.hash(&txid, vout), expected.as_slice());
            assert_eq!(session.finish(prepared, vout), expected.as_slice());
        }

        let session = Session::new(&[0xff; 16]);
        let txid = [0xff; 32];
        let expected = hex::decode("13758e1ba5e1dcc9e0838ebea54fa28b").unwrap();
        assert_eq!(session.hash(&txid, 0), expected.as_slice());
        assert_eq!(
            session.finish(session.prepare_txid(&txid), 0),
            expected.as_slice()
        );
    }
}

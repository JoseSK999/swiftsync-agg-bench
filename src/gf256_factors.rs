// SPDX-License-Identifier: MIT
//! OutPoint factors `z XOR txid XOR r * encoded_vout` for the GF256 aggregator.
//! A txid occupies all 256 polynomial coefficients, in serialized little-endian
//! byte order. Vout bit i is coefficient x^i, not an integer-addition counter.
//! Secrets z and r are independent 32-byte session values supplied by the caller.
//!
//! Direct mode computes r * vout. Cached modes reuse vouts 0/1 or a table for
//! vouts 0..255, with direct fallback. A four-byte table policy is also tested.
//! Session setup builds four 8 KiB tables and is excluded from timing. Only
//! public vout values select table entries, and no per-block allocation is needed.

use crate::gf256_product::Field;
use crate::gf256_product::{self as field};

pub const DIRECT: u8 = 0;
pub const COMMON256: u8 = 1;
pub const BYTE4: u8 = 2;
pub const FAST01: u8 = 3;
#[cfg(test)]
pub const COMMON_TABLE_BYTES: usize = 256 * 32;
#[cfg(test)]
pub const FULL_TABLE_BYTES: usize = 4 * COMMON_TABLE_BYTES;
#[cfg(test)]
pub const SECRET_BYTES: usize = 64;

pub struct Session {
    z: Field,
    r: Field,
    z_xor_r: Field,
    tables: [[Field; 256]; 4],
    accelerated: bool,
    #[cfg(any(test, feature = "software"))]
    reference: bool,
}

/// The canonical zero-extended polynomial encoding of an original output index.
#[cfg(test)]
#[inline(always)]
pub fn encoded_vout(vout: u32) -> Field {
    Field([u64::from(vout), 0, 0, 0])
}

impl Session {
    /// Prepares the vout tables and chooses the multiplication backend once.
    pub fn new(z: &[u8; 32], r: &[u8; 32]) -> Self {
        let z = Field::from_bytes(*z);
        let r = Field::from_bytes(*r);
        let mut basis = [Field::ZERO; 32];
        basis[0] = r;
        for i in 1..32 {
            basis[i] = basis[i - 1].times_x();
        }
        let mut tables = [[Field::ZERO; 256]; 4];
        for position in 0..4 {
            for value in 1usize..256 {
                let bit = value.trailing_zeros() as usize;
                tables[position][value] =
                    tables[position][value & (value - 1)].xor(basis[position * 8 + bit]);
            }
        }
        Self {
            z,
            r,
            z_xor_r: z.xor(r),
            tables,
            accelerated: field::is_accelerated(),
            #[cfg(any(test, feature = "software"))]
            reference: false,
        }
    }

    /// Forces the independent bit-at-a-time backend, including vout multiplication.
    #[cfg(any(test, feature = "software"))]
    pub fn new_reference(z: &[u8; 32], r: &[u8; 32]) -> Self {
        Self {
            accelerated: false,
            reference: true,
            ..Self::new(z, r)
        }
    }

    /// Prepares z XOR txid once for outputs sharing a transaction.
    #[inline(always)]
    pub fn prefix(&self, txid: &[u8; 32]) -> Field {
        self.z.xor(Field::from_bytes(*txid))
    }

    #[inline(always)]
    fn direct_vout(&self, vout: u32) -> Field {
        #[cfg(target_arch = "aarch64")]
        if self.accelerated {
            // SAFETY: Constructor checked the AES/PMULL feature for this session.
            return unsafe { field::mul_u32_pmull(self.r, vout) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.accelerated {
            // SAFETY: Constructor detected PCLMUL for this session.
            return unsafe { field::mul_u32_pclmul(self.r, vout) };
        }
        #[cfg(any(test, feature = "software"))]
        if self.reference {
            return field::scalar_mul_u32(self.r, vout);
        }
        field::mul_u32(self.r, vout)
    }

    /// Computes r * vout using the selected compile-time cache policy.
    #[inline(always)]
    pub fn vout_term<const CACHE: u8>(&self, vout: u32) -> Field {
        match CACHE {
            DIRECT => self.direct_vout(vout),
            FAST01 => match vout {
                0 => Field::ZERO,
                1 => self.r,
                _ => self.direct_vout(vout),
            },
            COMMON256 => {
                if vout < 256 {
                    self.tables[0][vout as usize]
                } else {
                    self.direct_vout(vout)
                }
            }
            BYTE4 => {
                let bytes = vout.to_le_bytes();
                self.tables[0][bytes[0] as usize]
                    .xor(self.tables[1][bytes[1] as usize])
                    .xor(self.tables[2][bytes[2] as usize])
                    .xor(self.tables[3][bytes[3] as usize])
            }
            _ => panic!("unknown GF256 vout cache policy"),
        }
    }

    #[inline(always)]
    pub fn finish<const CACHE: u8>(&self, prefix: Field, vout: u32) -> Field {
        prefix.xor(self.vout_term::<CACHE>(vout))
    }

    /// Computes the complete OutPoint factor, including its vout contribution.
    #[inline(always)]
    pub fn factor<const CACHE: u8>(&self, txid: &[u8; 32], vout: u32) -> Field {
        if CACHE == FAST01 {
            match vout {
                0 => self.z.xor(Field::from_bytes(*txid)),
                1 => self.z_xor_r.xor(Field::from_bytes(*txid)),
                _ => self.prefix(txid).xor(self.direct_vout(vout)),
            }
        } else {
            self.finish::<CACHE>(self.prefix(txid), vout)
        }
    }

    /// Computes a reference factor using bit-at-a-time field multiplication.
    #[cfg(test)]
    pub fn reference_factor(&self, txid: &[u8; 32], vout: u32) -> Field {
        self.z
            .xor(Field::from_bytes(*txid))
            .xor(field::scalar_mul(self.r, encoded_vout(vout)))
    }

    #[inline(always)]
    pub fn multiply(&self, a: Field, b: Field) -> Field {
        #[cfg(target_arch = "aarch64")]
        if self.accelerated {
            // SAFETY: Constructor checked AES/PMULL before saving this flag.
            return unsafe { field::mul_pmull(a, b) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.accelerated {
            // SAFETY: Constructor detected PCLMUL for this session.
            return unsafe { field::mul_pclmul(a, b) };
        }
        #[cfg(any(test, feature = "software"))]
        if self.reference {
            return field::scalar_mul(a, b);
        }
        #[cfg(feature = "software")]
        return crate::gf256_software::mul(a, b);
        #[cfg(not(feature = "software"))]
        field::scalar_mul(a, b)
    }

    /// Independent partial products using the session's selected backend.
    #[inline(always)]
    pub fn multiply_batch<const N: usize>(&self, a: [Field; N], b: [Field; N]) -> [Field; N] {
        #[cfg(target_arch = "aarch64")]
        if self.accelerated {
            // SAFETY: Constructor checked AES/PMULL before saving this flag.
            return unsafe { field::product_batch_pmull(a, b) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.accelerated {
            // SAFETY: Constructor detected PCLMUL for this session.
            return unsafe { field::product_batch_pclmul(a, b) };
        }
        #[cfg(any(test, feature = "software"))]
        if self.reference {
            return core::array::from_fn(|i| field::scalar_mul(a[i], b[i]));
        }
        #[cfg(feature = "software")]
        return core::array::from_fn(|i| crate::gf256_software::mul(a[i], b[i]));
        #[cfg(not(feature = "software"))]
        core::array::from_fn(|i| field::scalar_mul(a[i], b[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vout_is_a_zero_extended_polynomial_not_integer_repeated_addition() {
        let session = Session::new(&[0; 32], &Field::ONE.to_bytes());
        for vout in [
            0,
            1,
            2,
            3,
            255,
            256,
            65535,
            65536,
            0x01020304,
            0x80000000,
            u32::MAX,
        ] {
            let want = encoded_vout(vout);
            assert_eq!(session.vout_term::<DIRECT>(vout), want);
            assert_eq!(session.vout_term::<COMMON256>(vout), want);
            assert_eq!(session.vout_term::<BYTE4>(vout), want);
            assert_eq!(session.vout_term::<FAST01>(vout), want);
            let mut serialized = [0; 32];
            serialized[..4].copy_from_slice(&vout.to_le_bytes());
            assert_eq!(want.to_bytes(), serialized);
        }
        // 1 + 1 in this field is zero, but encoded vout 2 is the polynomial x.
        assert_eq!(Field::ONE.xor(Field::ONE), Field::ZERO);
        assert_ne!(encoded_vout(2), Field::ZERO);
    }

    #[test]
    fn cache_tables_and_factors_match_scalar_for_multiple_keys() {
        let mut seed = 0xa037d914be2695c1u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..8 {
            let z = core::array::from_fn(|_| next() as u8);
            let r = core::array::from_fn(|_| next() as u8);
            let session = Session::new(&z, &r);
            #[cfg(feature = "software")]
            assert!(!session.accelerated);
            for vout in (0..256).chain([256, 65535, 65536, 0x12345678, 0x80000000, u32::MAX]) {
                let txid = core::array::from_fn(|_| next() as u8);
                let want = session.reference_factor(&txid, vout);
                assert_eq!(session.factor::<DIRECT>(&txid, vout), want);
                assert_eq!(session.factor::<COMMON256>(&txid, vout), want);
                assert_eq!(session.factor::<BYTE4>(&txid, vout), want);
                assert_eq!(session.factor::<FAST01>(&txid, vout), want);
            }
            for _ in 0..64 {
                let vout = next() as u32;
                let txid = core::array::from_fn(|_| next() as u8);
                let want = session.reference_factor(&txid, vout);
                assert_eq!(session.factor::<DIRECT>(&txid, vout), want);
                assert_eq!(session.factor::<COMMON256>(&txid, vout), want);
                assert_eq!(session.factor::<BYTE4>(&txid, vout), want);
                assert_eq!(session.factor::<FAST01>(&txid, vout), want);
            }
        }
        assert_eq!(COMMON_TABLE_BYTES, 8192);
        assert_eq!(FULL_TABLE_BYTES, 32768);
        assert_eq!(SECRET_BYTES, 64);
    }
}

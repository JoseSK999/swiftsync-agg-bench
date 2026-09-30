// SPDX-License-Identifier: MIT
//! Portable GF(2^256) multiplication without carry-less multiplication instructions.
//!
//! Integer multiplication with bit holes computes each 64-bit polynomial product.
//! Nested Karatsuba needs nine such products. The sparse modulus then reduces the
//! 512-bit result using only shifts and XORs. This preserves the representation and
//! modulus of `gf256_product`, whose bit-at-a-time implementation remains the
//! independent reference.
//!
//! The hole technique is described at <https://bearssl.org/constanttime.html>.
//! All loops, memory accesses and shifts have public, fixed bounds, with no
//! secret-dependent lookups. Constant-time use also requires the host CPU's
//! ordinary integer multiplications to be constant-time.

use super::gf256_product::Field;

/// Four-bit spacing leaves three carry bits between polynomial coefficients.
const HOLES: u64 = 0x1111_1111_1111_1111;
const WIDE_HOLES: u128 = 0x1111_1111_1111_1111_1111_1111_1111_1111;

/// Multiplies sparse lanes and extracts the parity of each integer coefficient.
///
/// At least one operand must have at most fifteen bits in every lane. Then no
/// integer product adds sixteen terms at a coefficient, crossing a carry hole.
#[inline(always)]
fn holes_product(a: u64, b: u64) -> u128 {
    let a = core::array::from_fn::<_, 4, _>(|i| a & (HOLES << i));
    let b = core::array::from_fn::<_, 4, _>(|i| b & (HOLES << i));
    let mut lanes = [0_u128; 4];
    for (i, left) in a.into_iter().enumerate() {
        for (j, right) in b.into_iter().enumerate() {
            lanes[(i + j) & 3] ^= u128::from(left) * u128::from(right);
        }
    }
    (lanes[0] & WIDE_HOLES)
        ^ (lanes[1] & (WIDE_HOLES << 1))
        ^ (lanes[2] & (WIDE_HOLES << 2))
        ^ (lanes[3] & (WIDE_HOLES << 3))
}

/// Full carry-less 64-by-64 product, including coefficients above bit 63.
#[inline(always)]
fn carryless64(a: u64, b: u64) -> u128 {
    // A complete lane has sixteen bits. Removing the bottom nibble limits it
    // to fifteen, preventing carries from corrupting the next retained bit.
    // Restore those four bits with branch-free polynomial shifts of b.
    let mut result = holes_product(a & !15, b);
    for bit in 0..4 {
        let mask = 0_u64.wrapping_sub((a >> bit) & 1);
        result ^= u128::from(b & mask) << bit;
    }
    result
}

/// Converts polynomial coefficients into two little-endian 64-bit limbs.
#[inline(always)]
fn limbs(value: u128) -> [u64; 2] {
    let bytes = value.to_le_bytes();
    [
        u64::from_le_bytes(bytes[..8].try_into().expect("exactly eight bytes")),
        u64::from_le_bytes(bytes[8..].try_into().expect("exactly eight bytes")),
    ]
}

/// Karatsuba's 128-by-128 product needs three 64-by-64 products.
#[inline(always)]
fn product128(a: [u64; 2], b: [u64; 2]) -> [u64; 4] {
    let low = carryless64(a[0], b[0]);
    let high = carryless64(a[1], b[1]);
    let middle = carryless64(a[0] ^ a[1], b[0] ^ b[1]) ^ low ^ high;
    let low = limbs(low);
    let high = limbs(high);
    let middle = limbs(middle);
    [low[0], low[1] ^ middle[0], high[0] ^ middle[1], high[1]]
}

/// Low and overflowing parts of multiplication by 1 + x^2 + x^5 + x^10.
#[inline(always)]
fn modulus_product(word: u64) -> [u64; 2] {
    [
        word ^ (word << 2) ^ (word << 5) ^ (word << 10),
        (word >> 62) ^ (word >> 59) ^ (word >> 54),
    ]
}

/// Folds the upper half at x^256, then folds its at-most-ten-bit overflow.
#[inline(always)]
fn reduce(product: [u64; 8]) -> Field {
    let [l0, l1, l2, l3, h0, h1, h2, h3] = product;
    let q3 = modulus_product(h3);
    // The second fold fits in the bottom twenty bits. Incorporating it into
    // h0 before the first fold saves a separate multiplication by the modulus.
    let q0 = modulus_product(h0 ^ q3[1]);
    let q1 = modulus_product(h1);
    let q2 = modulus_product(h2);
    Field([
        l0 ^ q0[0],
        l1 ^ q1[0] ^ q0[1],
        l2 ^ q2[0] ^ q1[1],
        l3 ^ q3[0] ^ q2[1],
    ])
}

/// Multiplies two field elements using nested Karatsuba and sparse reduction.
#[inline]
pub(crate) fn mul(a: Field, b: Field) -> Field {
    let [a0, a1, a2, a3] = a.0;
    let [b0, b1, b2, b3] = b.0;
    let low = product128([a0, a1], [b0, b1]);
    let high = product128([a2, a3], [b2, b3]);
    let middle = product128([a0 ^ a2, a1 ^ a3], [b0 ^ b2, b1 ^ b3]);
    let middle: [u64; 4] = core::array::from_fn(|i| middle[i] ^ low[i] ^ high[i]);
    reduce([
        low[0],
        low[1],
        low[2] ^ middle[0],
        low[3] ^ middle[1],
        high[0] ^ middle[2],
        high[1] ^ middle[3],
        high[2],
        high[3],
    ])
}

/// Multiplies by a zero-extended vout with four 64-by-32 products.
#[inline]
pub(crate) fn mul_u32(a: Field, b: u32) -> Field {
    // b contributes at most eight bits per sparse lane, so holes_product is
    // carry-safe without removing or separately restoring a's bottom nibble.
    let products = a.0.map(|word| limbs(holes_product(word, u64::from(b))));
    reduce([
        products[0][0],
        products[1][0] ^ products[0][1],
        products[2][0] ^ products[1][1],
        products[3][0] ^ products[2][1],
        products[3][1],
        0,
        0,
        0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf256_product::scalar_mul;

    /// Deterministic test data, unrelated to benchmark session secrets.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn basis(bit: usize) -> Field {
        let mut limbs = [0; 4];
        limbs[bit / 64] = 1 << (bit % 64);
        Field(limbs)
    }

    #[test]
    fn carryless_product_handles_every_basis_pair_and_dense_words() {
        for i in 0..64 {
            for j in 0..64 {
                assert_eq!(carryless64(1 << i, 1 << j), 1_u128 << (i + j));
            }
        }
        let mut state = 0x4c12_8705_03b2_d561;
        let patterns = [0, 1, u64::MAX, HOLES, !HOLES, 0x8000_0000_0000_0000];
        for a in patterns
            .into_iter()
            .chain((0..1024).map(|_| next(&mut state)))
        {
            for b in patterns {
                let mut expected = 0;
                for bit in 0..64 {
                    let mask = 0_u128.wrapping_sub(u128::from((b >> bit) & 1));
                    expected ^= (u128::from(a) << bit) & mask;
                }
                assert_eq!(carryless64(a, b), expected);
            }
        }
    }

    #[test]
    fn sparse_reducer_handles_every_product_bit() {
        let mut expected = Field::ONE;
        for bit in 0..512 {
            let mut product = [0; 8];
            product[bit / 64] = 1 << (bit % 64);
            assert_eq!(reduce(product), expected);
            expected = expected.times_x();
        }
    }

    #[test]
    fn field_product_handles_every_basis_pair() {
        let mut powers = [Field::ZERO; 511];
        powers[0] = Field::ONE;
        for bit in 1..powers.len() {
            powers[bit] = powers[bit - 1].times_x();
        }
        for i in 0..256 {
            for j in 0..256 {
                assert_eq!(mul(basis(i), basis(j)), powers[i + j]);
            }
        }
    }

    #[test]
    fn dense_boundary_and_random_products_match_bitwise_reference() {
        let patterns = [
            Field::ZERO,
            Field::ONE,
            Field([u64::MAX; 4]),
            Field([HOLES; 4]),
            Field([!HOLES; 4]),
            basis(63),
            basis(64),
            basis(127),
            basis(128),
            basis(191),
            basis(192),
            basis(255),
        ];
        for a in patterns {
            for b in patterns {
                assert_eq!(mul(a, b), scalar_mul(a, b));
            }
        }
        let mut state = 0x567f_0568_1539_81ae;
        for _ in 0..4096 {
            let a = Field(core::array::from_fn(|_| next(&mut state)));
            let b = Field(core::array::from_fn(|_| next(&mut state)));
            assert_eq!(mul(a, b), scalar_mul(a, b));
        }
    }

    #[test]
    fn vout_products_match_bitwise_reference() {
        let mut state = 0x1779_0c62_45af_dc03;
        let mut values = vec![0, 1, 2, 3, u32::MAX, 0xaaaa_aaaa, 0x5555_5555];
        values.extend((0..32).map(|bit| 1 << bit));
        values.extend((0..1024).map(|_| {
            let bytes = next(&mut state).to_le_bytes();
            u32::from_le_bytes(bytes[..4].try_into().unwrap())
        }));
        for b in values {
            for a in [
                Field([u64::MAX; 4]),
                Field(core::array::from_fn(|_| next(&mut state))),
            ] {
                assert_eq!(mul_u32(a, b), scalar_mul(a, Field([u64::from(b), 0, 0, 0])));
            }
        }
    }
}

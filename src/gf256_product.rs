// SPDX-License-Identifier: MIT
//! GF(2^256) arithmetic for the product aggregator, using PMULL or PCLMULQDQ.
//!
//! Bit i of the little-endian encoding is the coefficient of x^i. Reduction is
//! modulo x^256 + x^10 + x^5 + x^2 + 1. Independent vectors and a Rabin
//! irreducibility certificate are stored in `gf256-reference/vectors.json`.
//! A bit-at-a-time implementation provides the test and benchmark reference.

/// Low coefficients of the irreducible modulus; x^256 is implicit.
pub const MODULUS_LOW: u64 = 0x425;

/// Four little-endian polynomial limbs, including all 256 bits of a txid.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Field(pub [u64; 4]);

impl Field {
    pub const ZERO: Self = Self([0; 4]);
    pub const ONE: Self = Self([1, 0, 0, 0]);

    #[inline]
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(core::array::from_fn(|i| {
            u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())
        }))
    }

    #[cfg(test)]
    #[inline]
    pub fn to_bytes(self) -> [u8; 32] {
        let mut bytes = [0; 32];
        for (word, chunk) in self.0.into_iter().zip(bytes.chunks_exact_mut(8)) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    #[inline]
    pub fn xor(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] ^ rhs.0[i]))
    }

    /// Multiplication by x, including its possible reduction at degree 256.
    #[inline]
    pub fn times_x(self) -> Self {
        let [a, b, c, d] = self.0;
        Self([
            (a << 1) ^ (MODULUS_LOW & 0_u64.wrapping_sub(d >> 63)),
            (b << 1) | (a >> 63),
            (c << 1) | (b >> 63),
            (d << 1) | (c >> 63),
        ])
    }
}

/// Independent bit-at-a-time reference; intentionally not a fast fallback.
pub fn scalar_mul(mut a: Field, b: Field) -> Field {
    let mut result = Field::ZERO;
    for word in b.0 {
        for bit in 0..64 {
            let mask = 0_u64.wrapping_sub((word >> bit) & 1);
            for (out, input) in result.0.iter_mut().zip(a.0) {
                *out ^= input & mask;
            }
            a = a.times_x();
        }
    }
    result
}

/// Whether this build can use the CPU's carry-less multiplication instructions.
#[inline]
pub fn is_accelerated() -> bool {
    // Forced software disables PMULL/PCLMUL even on capable CPUs.
    if cfg!(feature = "software") {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("aes")
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("pclmulqdq")
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// Dispatches field multiplication to hardware or the optimized software kernel.
/// ARM PMULL is detected through Rust's `aes` feature.
#[cfg(test)]
#[inline]
pub fn mul(a: Field, b: Field) -> Field {
    #[cfg(target_arch = "aarch64")]
    if is_accelerated() {
        // SAFETY: The feature check covers every PMULL instruction below.
        return unsafe { mul_pmull(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_accelerated() {
        // SAFETY: PCLMUL was detected before entering the x86 kernel.
        return unsafe { x86::mul_pclmul(a, b) };
    }
    crate::gf256_software::mul(a, b)
}

/// Dispatches independent product updates to the selected multiplication backend.
#[cfg(test)]
#[inline]
pub fn product_batch<const N: usize>(a: [Field; N], b: [Field; N]) -> [Field; N] {
    #[cfg(target_arch = "aarch64")]
    if is_accelerated() {
        // SAFETY: The feature check covers every PMULL instruction below.
        return unsafe { product_batch_pmull(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_accelerated() {
        // SAFETY: PCLMUL was detected before entering the x86 kernel.
        return unsafe { x86::product_batch_pclmul(a, b) };
    }
    core::array::from_fn(|i| crate::gf256_software::mul(a[i], b[i]))
}

/// Multiplies by a zero-extended vout without a general 256-by-256 product.
#[inline]
pub fn mul_u32(a: Field, b: u32) -> Field {
    #[cfg(target_arch = "aarch64")]
    if is_accelerated() {
        // SAFETY: The feature check covers every PMULL instruction below.
        return unsafe { mul_u32_pmull(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    if is_accelerated() {
        // SAFETY: PCLMUL was detected before entering the x86 kernel.
        return unsafe { x86::mul_u32_pclmul(a, b) };
    }
    #[cfg(feature = "software")]
    return crate::gf256_software::mul_u32(a, b);
    #[cfg(not(feature = "software"))]
    scalar_mul_u32(a, b)
}

/// Independent bit-at-a-time vout multiplication for reference benchmarks.
pub fn scalar_mul_u32(a: Field, b: u32) -> Field {
    let mut result = Field::ZERO;
    let mut shifted = a;
    for bit in 0..32 {
        let mask = 0_u64.wrapping_sub(u64::from((b >> bit) & 1));
        for (out, input) in result.0.iter_mut().zip(shifted.0) {
            *out ^= input & mask;
        }
        shifted = shifted.times_x();
    }
    result
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::*;
    use core::mem::transmute;

    use super::Field;
    use super::MODULUS_LOW;

    #[derive(Clone, Copy)]
    struct Limbs {
        low: uint64x2_t,
        high: uint64x2_t,
    }

    #[derive(Clone, Copy)]
    struct Raw128 {
        low: uint64x2_t,
        high: uint64x2_t,
    }

    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn low_mul(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // SAFETY: Caller enables PMULL; this is only a 128-bit reinterpretation.
        unsafe { transmute(vmull_p64(vgetq_lane_u64::<0>(a), vgetq_lane_u64::<0>(b))) }
    }

    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn high_mul(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // SAFETY: Caller enables PMULL; this is only a 128-bit reinterpretation.
        unsafe {
            transmute(vmull_high_p64(
                vreinterpretq_p64_u64(a),
                vreinterpretq_p64_u64(b),
            ))
        }
    }

    /// Nested Karatsuba's inner 128-by-128 product uses three PMULLs.
    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn raw128(a: uint64x2_t, b: uint64x2_t) -> Raw128 {
        // SAFETY: Caller enables PMULL; all vector operations have valid lanes.
        unsafe {
            let zero = vdupq_n_u64(0);
            let low = low_mul(a, b);
            let high = high_mul(a, b);
            let folded_a = veorq_u64(a, vextq_u64::<1>(a, a));
            let folded_b = veorq_u64(b, vextq_u64::<1>(b, b));
            let middle = veorq_u64(low_mul(folded_a, folded_b), veorq_u64(low, high));
            Raw128 {
                low: veorq_u64(low, vextq_u64::<1>(zero, middle)),
                high: veorq_u64(high, vextq_u64::<1>(middle, zero)),
            }
        }
    }

    /// Reduces a 512-bit product with four further carry-less multiplies.
    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn reduce(low: Limbs, high: Limbs) -> Limbs {
        // Write high as four 64-bit limbs h0..h3, and q=0x425. Each limb
        // contributes hi*q at its original position. h3*q's upper limb is the
        // only overflow; folding it into h0 before h0*q reduces it for free.
        // This overflow has at most ten bits, so its q-product fits in 64 bits.
        // SAFETY: Caller enables PMULL; lane extraction only rearranges vectors.
        unsafe {
            let zero = vdupq_n_u64(0);
            let modulus = vdupq_n_u64(MODULUS_LOW);
            let h3 = high_mul(high.high, modulus);
            let h0_input = veorq_u64(high.low, vextq_u64::<1>(h3, zero));
            let h0 = low_mul(h0_input, modulus);
            let h1 = high_mul(high.low, modulus);
            let h2 = low_mul(high.high, modulus);
            Limbs {
                low: veorq_u64(veorq_u64(low.low, h0), vextq_u64::<1>(zero, h1)),
                high: veorq_u64(veorq_u64(low.high, h2), vextq_u64::<1>(h1, h3)),
            }
        }
    }

    #[inline]
    unsafe fn load(value: Field) -> Limbs {
        // SAFETY: Both loads address two initialized u64 limbs.
        unsafe {
            Limbs {
                low: vld1q_u64(value.0.as_ptr()),
                high: vld1q_u64(value.0.as_ptr().add(2)),
            }
        }
    }

    #[inline]
    unsafe fn store(value: Limbs) -> Field {
        let mut result = Field::ZERO;
        // SAFETY: Both stores address two valid output u64 limbs.
        unsafe {
            vst1q_u64(result.0.as_mut_ptr(), value.low);
            vst1q_u64(result.0.as_mut_ptr().add(2), value.high);
        }
        result
    }

    /// Updates N independent field products with 9 Karatsuba + 4 reduction PMULLs.
    ///
    /// # Safety
    /// The CPU must support ARM AES/PMULL. Feature detection belongs outside hot loops.
    #[target_feature(enable = "aes")]
    #[inline]
    pub unsafe fn product_batch_pmull<const N: usize>(a: [Field; N], b: [Field; N]) -> [Field; N] {
        // SAFETY: Caller enables PMULL; all arrays are initialized for exactly N lanes.
        unsafe {
            let a = a.map(|value| load(value));
            let b = b.map(|value| load(value));
            let low: [Raw128; N] = core::array::from_fn(|i| raw128(a[i].low, b[i].low));
            let high: [Raw128; N] = core::array::from_fn(|i| raw128(a[i].high, b[i].high));
            let folded: [Raw128; N] = core::array::from_fn(|i| {
                raw128(
                    veorq_u64(a[i].low, a[i].high),
                    veorq_u64(b[i].low, b[i].high),
                )
            });
            core::array::from_fn(|i| {
                let middle_low = veorq_u64(folded[i].low, veorq_u64(low[i].low, high[i].low));
                let middle_high = veorq_u64(folded[i].high, veorq_u64(low[i].high, high[i].high));
                store(reduce(
                    Limbs {
                        low: low[i].low,
                        high: veorq_u64(low[i].high, middle_low),
                    },
                    Limbs {
                        low: veorq_u64(high[i].low, middle_high),
                        high: high[i].high,
                    },
                ))
            })
        }
    }

    /// One general field product; callers should interleave independent accumulators.
    ///
    /// # Safety
    /// The CPU must support ARM AES/PMULL.
    #[target_feature(enable = "aes")]
    #[inline]
    pub unsafe fn mul_pmull(a: Field, b: Field) -> Field {
        // SAFETY: Caller enables PMULL.
        unsafe { product_batch_pmull([a], [b])[0] }
    }

    /// Four limb products plus one overflow fold for a 32-bit multiplier.
    ///
    /// # Safety
    /// The CPU must support ARM AES/PMULL.
    #[target_feature(enable = "aes")]
    #[inline]
    pub unsafe fn mul_u32_pmull(a: Field, b: u32) -> Field {
        // SAFETY: Caller enables PMULL; both field loads stay within four limbs.
        unsafe {
            let a = load(a);
            let b = vdupq_n_u64(u64::from(b));
            let zero = vdupq_n_u64(0);
            let p0 = low_mul(a.low, b);
            let p1 = high_mul(a.low, b);
            let p2 = low_mul(a.high, b);
            let p3 = high_mul(a.high, b);
            let overflow = high_mul(p3, vdupq_n_u64(MODULUS_LOW));
            store(Limbs {
                low: veorq_u64(veorq_u64(p0, vextq_u64::<1>(zero, p1)), overflow),
                high: veorq_u64(p2, vextq_u64::<1>(p1, p3)),
            })
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use arm::mul_pmull;
#[cfg(target_arch = "aarch64")]
pub use arm::mul_u32_pmull;
#[cfg(target_arch = "aarch64")]
pub use arm::product_batch_pmull;

/// The same 9-product Karatsuba and 4-product reduction as the ARM kernel.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    use super::Field;
    use super::MODULUS_LOW;

    #[derive(Clone, Copy)]
    struct Limbs {
        low: __m128i,
        high: __m128i,
    }

    #[derive(Clone, Copy)]
    struct Raw128 {
        low: __m128i,
        high: __m128i,
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    unsafe fn low_mul(a: __m128i, b: __m128i) -> __m128i {
        _mm_clmulepi64_si128(a, b, 0x00)
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    unsafe fn high_mul(a: __m128i, b: __m128i) -> __m128i {
        _mm_clmulepi64_si128(a, b, 0x11)
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    unsafe fn raw128(a: __m128i, b: __m128i) -> Raw128 {
        // SAFETY: Caller detected PCLMUL; SSE2 is baseline on x86-64.
        unsafe {
            let low = low_mul(a, b);
            let high = high_mul(a, b);
            let folded_a = _mm_xor_si128(a, _mm_srli_si128(a, 8));
            let folded_b = _mm_xor_si128(b, _mm_srli_si128(b, 8));
            let middle = _mm_xor_si128(low_mul(folded_a, folded_b), _mm_xor_si128(low, high));
            Raw128 {
                low: _mm_xor_si128(low, _mm_slli_si128(middle, 8)),
                high: _mm_xor_si128(high, _mm_srli_si128(middle, 8)),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    unsafe fn reduce(low: Limbs, high: Limbs) -> Limbs {
        // SAFETY: Caller detected PCLMUL; q=x^10+x^5+x^2+1.
        unsafe {
            let modulus = _mm_set1_epi64x(i64::try_from(MODULUS_LOW).unwrap());
            let h3 = high_mul(high.high, modulus);
            let h0 = low_mul(_mm_xor_si128(high.low, _mm_srli_si128(h3, 8)), modulus);
            let h1 = high_mul(high.low, modulus);
            let h2 = low_mul(high.high, modulus);
            Limbs {
                low: _mm_xor_si128(_mm_xor_si128(low.low, h0), _mm_slli_si128(h1, 8)),
                high: _mm_xor_si128(
                    _mm_xor_si128(low.high, h2),
                    _mm_xor_si128(_mm_srli_si128(h1, 8), _mm_slli_si128(h3, 8)),
                ),
            }
        }
    }

    #[inline]
    unsafe fn load(value: Field) -> Limbs {
        // SAFETY: Both unaligned loads address two initialized u64 limbs.
        unsafe {
            Limbs {
                low: _mm_loadu_si128(value.0.as_ptr().cast()),
                high: _mm_loadu_si128(value.0.as_ptr().add(2).cast()),
            }
        }
    }

    #[inline]
    unsafe fn store(value: Limbs) -> Field {
        let mut result = Field::ZERO;
        // SAFETY: Both unaligned stores address two output u64 limbs.
        unsafe {
            _mm_storeu_si128(result.0.as_mut_ptr().cast(), value.low);
            _mm_storeu_si128(result.0.as_mut_ptr().add(2).cast(), value.high);
        }
        result
    }

    /// # Safety
    /// The CPU must support PCLMULQDQ.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub unsafe fn product_batch_pclmul<const N: usize>(a: [Field; N], b: [Field; N]) -> [Field; N] {
        // SAFETY: Caller detected PCLMUL; each array contains N initialized fields.
        unsafe {
            let a = a.map(|value| load(value));
            let b = b.map(|value| load(value));
            let low: [Raw128; N] = core::array::from_fn(|i| raw128(a[i].low, b[i].low));
            let high: [Raw128; N] = core::array::from_fn(|i| raw128(a[i].high, b[i].high));
            let folded: [Raw128; N] = core::array::from_fn(|i| {
                raw128(
                    _mm_xor_si128(a[i].low, a[i].high),
                    _mm_xor_si128(b[i].low, b[i].high),
                )
            });
            core::array::from_fn(|i| {
                let middle_low =
                    _mm_xor_si128(folded[i].low, _mm_xor_si128(low[i].low, high[i].low));
                let middle_high =
                    _mm_xor_si128(folded[i].high, _mm_xor_si128(low[i].high, high[i].high));
                store(reduce(
                    Limbs {
                        low: low[i].low,
                        high: _mm_xor_si128(low[i].high, middle_low),
                    },
                    Limbs {
                        low: _mm_xor_si128(high[i].low, middle_high),
                        high: high[i].high,
                    },
                ))
            })
        }
    }

    /// # Safety
    /// The CPU must support PCLMULQDQ.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub unsafe fn mul_pclmul(a: Field, b: Field) -> Field {
        // SAFETY: Caller detected PCLMUL.
        unsafe { product_batch_pclmul([a], [b])[0] }
    }

    /// # Safety
    /// The CPU must support PCLMULQDQ.
    #[inline]
    #[target_feature(enable = "pclmulqdq")]
    pub unsafe fn mul_u32_pclmul(a: Field, b: u32) -> Field {
        // SAFETY: Caller detected PCLMUL.
        unsafe {
            let a = load(a);
            let b = _mm_set1_epi64x(i64::from(b));
            let modulus = _mm_set1_epi64x(i64::try_from(MODULUS_LOW).unwrap());
            let p0 = low_mul(a.low, b);
            let p1 = high_mul(a.low, b);
            let p2 = low_mul(a.high, b);
            let p3 = high_mul(a.high, b);
            let overflow = high_mul(p3, modulus);
            store(Limbs {
                low: _mm_xor_si128(_mm_xor_si128(p0, _mm_slli_si128(p1, 8)), overflow),
                high: _mm_xor_si128(
                    p2,
                    _mm_xor_si128(_mm_srli_si128(p1, 8), _mm_slli_si128(p3, 8)),
                ),
            })
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::mul_pclmul;
#[cfg(target_arch = "x86_64")]
pub use x86::mul_u32_pclmul;
#[cfg(target_arch = "x86_64")]
pub use x86::product_batch_pclmul;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_integer_polynomial_vectors() {
        let vectors: serde_json::Value =
            serde_json::from_str(include_str!("../gf256-reference/vectors.json")).unwrap();
        fn field(value: &serde_json::Value) -> Field {
            let hex = value.as_str().unwrap();
            assert_eq!(hex.len(), 64);
            Field::from_bytes(core::array::from_fn(|i| {
                u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap()
            }))
        }
        assert_eq!(vectors["certificate"]["irreducible"], true);
        for case in vectors["multiplication"].as_array().unwrap() {
            let a = field(&case["a"]);
            let b = field(&case["b"]);
            let expected = field(&case["product"]);
            assert_eq!(scalar_mul(a, b), expected);
            assert_eq!(crate::gf256_software::mul(a, b), expected);
            assert_eq!(mul(a, b), expected);
        }
    }

    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn encoding_and_reduction_boundary() {
        let bytes = core::array::from_fn(|i| i as u8);
        assert_eq!(Field::from_bytes(bytes).to_bytes(), bytes);
        assert_eq!(
            Field([0, 0, 0, 1 << 63]).times_x(),
            Field([MODULUS_LOW, 0, 0, 0])
        );
        assert_eq!(mul(Field::ZERO, Field([u64::MAX; 4])), Field::ZERO);
        assert_eq!(mul(Field::ONE, Field([u64::MAX; 4])), Field([u64::MAX; 4]));
    }

    #[test]
    fn every_basis_bit_and_limb_boundary_match_reference() {
        for bit in 0..256 {
            let mut a = Field::ZERO;
            a.0[bit / 64] = 1 << (bit % 64);
            for boundary in [0, 1, 63, 64, 127, 128, 191, 192, 254, 255] {
                let mut b = Field::ZERO;
                b.0[boundary / 64] = 1 << (boundary % 64);
                assert_eq!(mul(a, b), scalar_mul(a, b), "bits {bit}, {boundary}");
            }
        }
    }

    #[test]
    fn random_products_u32_and_batch_sizes_match_reference() {
        #[cfg(feature = "software")]
        assert!(!is_accelerated());
        let mut state = 0x45af_390c_a457_8123;
        for _ in 0..512 {
            let a = Field(core::array::from_fn(|_| next(&mut state)));
            let b = Field(core::array::from_fn(|_| next(&mut state)));
            assert_eq!(mul(a, b), scalar_mul(a, b));
            let vout = next(&mut state) as u32;
            assert_eq!(
                mul_u32(a, vout),
                scalar_mul(a, Field([u64::from(vout), 0, 0, 0]))
            );
        }
        fn check<const N: usize>(state: &mut u64) {
            let a = core::array::from_fn(|_| Field(core::array::from_fn(|_| next(state))));
            let b = core::array::from_fn(|_| Field(core::array::from_fn(|_| next(state))));
            let actual = product_batch::<N>(a, b);
            for i in 0..N {
                assert_eq!(actual[i], scalar_mul(a[i], b[i]));
            }
        }
        check::<0>(&mut state);
        check::<1>(&mut state);
        check::<2>(&mut state);
        check::<3>(&mut state);
        check::<4>(&mut state);
        check::<8>(&mut state);
        check::<16>(&mut state);
    }

    #[test]
    fn field_laws_and_partitioned_products() {
        let mut state = 0x1345_daca_571e_0231;
        for _ in 0..128 {
            let a = Field(core::array::from_fn(|_| next(&mut state)));
            let b = Field(core::array::from_fn(|_| next(&mut state)));
            let c = Field(core::array::from_fn(|_| next(&mut state)));
            assert_eq!(mul(a, b), mul(b, a));
            assert_eq!(mul(mul(a, b), c), mul(a, mul(b, c)));
            assert_eq!(mul(a, b.xor(c)), mul(a, b).xor(mul(a, c)));
        }
        let values: [Field; 71] =
            core::array::from_fn(|_| Field(core::array::from_fn(|_| next(&mut state))));
        let serial = values.into_iter().fold(Field::ONE, mul);
        let mut lanes = [Field::ONE; 8];
        for (i, value) in values.into_iter().enumerate() {
            lanes[i % 8] = mul(lanes[i % 8], value);
        }
        assert_eq!(serial, lanes.into_iter().fold(Field::ONE, mul));
    }
}

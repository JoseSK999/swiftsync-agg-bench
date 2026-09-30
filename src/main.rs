// SPDX-License-Identifier: MIT

//! Compares OutPoint aggregators over two verified historical blocks.
//! Each timed traversal aggregates non-coinbase inputs and UTXO-eligible outputs
//! separately, including prefix preparation, batch tails and final merges.
//! Block decoding, output filtering, txid calculation and key setup are untimed.

mod cmac;
mod gf256_factors;
mod gf256_product;
#[cfg(any(test, feature = "software"))]
mod gf256_software;
mod sha256_tag;
mod sip128;
#[cfg(target_arch = "aarch64")]
mod sip128_neon;

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use bitcoin::Block;
use bitcoin::consensus::deserialize;
use bitcoin::hashes::Hash as _;

use crate::gf256_factors::COMMON256;
use crate::gf256_factors::DIRECT;
use crate::gf256_factors::FAST01;
use crate::gf256_factors::Session as GfSession;
use crate::gf256_product::Field;

const BLOCK_367891: &[u8] = include_bytes!("../fixtures/block_367891/raw.zst");
const BLOCK_866342: &[u8] = include_bytes!("../fixtures/block_866342/raw.zst");

#[derive(Clone)]
struct OutputGroup {
    txid: [u8; 32],
    vouts: Vec<u32>,
}

struct Workload {
    height: u32,
    block: Block,
    groups: Vec<OutputGroup>,
    inputs: usize,
    outputs: usize,
}

impl Workload {
    fn load(height: u32, compressed: &[u8], expected_hash: &str) -> Self {
        let raw = zstd::decode_all(compressed).expect("decode historical block");
        let block: Block = deserialize(&raw).expect("parse historical block");
        assert_eq!(block.block_hash().to_string(), expected_hash);
        assert!(block.check_merkle_root());
        let inputs = block.txdata.iter().skip(1).map(|tx| tx.input.len()).sum();
        let groups: Vec<_> = block
            .txdata
            .iter()
            .map(|tx| OutputGroup {
                txid: tx.compute_txid().to_byte_array(),
                vouts: tx
                    .output
                    .iter()
                    .enumerate()
                    .filter(|(_, output)| {
                        output.script_pubkey.len() <= 10_000 && !output.script_pubkey.is_op_return()
                    })
                    .map(|(vout, _)| u32::try_from(vout).expect("vout fits u32"))
                    .collect(),
            })
            .collect();
        let outputs = groups.iter().map(|group| group.vouts.len()).sum();
        Self {
            height,
            block,
            groups,
            inputs,
            outputs,
        }
    }
}

/// Separate sums of the two 64-bit tag halves, retaining carries in each u128.
/// Neither sum overflows with up to 2^64 records per side.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WideSum {
    low: u128,
    high: u128,
}

impl WideSum {
    #[inline]
    fn add_halves(&mut self, low: u64, high: u64) {
        self.low = self.low.wrapping_add(u128::from(low));
        self.high = self.high.wrapping_add(u128::from(high));
    }

    #[inline]
    fn add_tag(&mut self, tag: [u8; 16]) {
        self.add_halves(
            u64::from_le_bytes(tag[..8].try_into().expect("lower tag half")),
            u64::from_le_bytes(tag[8..].try_into().expect("upper tag half")),
        );
    }
}

/// Aggregates 128-bit SHA256 tags separately for added and removed OutPoints.
#[inline(never)]
fn sha256_block(work: &Workload, secret: &[u8; 16]) -> (WideSum, WideSum) {
    let mut removed = WideSum::default();
    for tx in work.block.txdata.iter().skip(1) {
        for input in &tx.input {
            let point = &input.previous_output;
            removed.add_tag(sha256_tag::hash(
                secret,
                point.txid.as_byte_array(),
                point.vout,
            ));
        }
    }

    let mut added = WideSum::default();
    for group in &work.groups {
        for &vout in &group.vouts {
            added.add_tag(sha256_tag::hash(secret, &group.txid, vout));
        }
    }
    (added, removed)
}

#[inline(never)]
fn cmac_block(work: &Workload, session: &cmac::Session) -> (WideSum, WideSum) {
    let mut removed = WideSum::default();
    for tx in work.block.txdata.iter().skip(1) {
        for input in &tx.input {
            let point = &input.previous_output;
            removed.add_tag(session.hash(point.txid.as_byte_array(), point.vout));
        }
    }

    let mut added = WideSum::default();
    for group in &work.groups {
        match group.vouts.as_slice() {
            [] => {}
            &[vout] => added.add_tag(session.hash(&group.txid, vout)),
            vouts => {
                let prefix = session.prepare_txid(&group.txid);
                for &vout in vouts {
                    added.add_tag(session.finish(prefix, vout));
                }
            }
        }
    }
    (added, removed)
}

#[inline(never)]
fn sip128_block(work: &Workload, keys: [u64; 2]) -> (WideSum, WideSum) {
    let mut removed = WideSum::default();
    for tx in work.block.txdata.iter().skip(1) {
        for input in &tx.input {
            let point = &input.previous_output;
            let (low, high) = sip128::hash(keys, point.txid.as_byte_array(), point.vout);
            removed.add_halves(low, high);
        }
    }

    let mut added = WideSum::default();
    for group in &work.groups {
        match group.vouts.as_slice() {
            [] => {}
            &[vout] => {
                let (low, high) = sip128::hash(keys, &group.txid, vout);
                added.add_halves(low, high);
            }
            vouts => {
                let prefix = sip128::TxidPrefix::new(keys, &group.txid);
                for &vout in vouts {
                    let (low, high) = prefix.finish(vout);
                    added.add_halves(low, high);
                }
            }
        }
    }
    (added, removed)
}

/// Aggregates four-record SIMD batches, reusing output prefixes and handling scalar tails.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn sip128_neon_block(work: &Workload, keys: [u64; 2]) -> (WideSum, WideSum) {
    let mut removed = WideSum::default();
    let zero = [0; 32];
    let mut txids = [&zero; 4];
    let mut vouts = [0; 4];
    let mut len = 0;
    for tx in work.block.txdata.iter().skip(1) {
        for input in &tx.input {
            txids[len] = input.previous_output.txid.as_byte_array();
            vouts[len] = input.previous_output.vout;
            len += 1;
            if len == 4 {
                for (low, high) in sip128_neon::hash(keys, txids, vouts) {
                    removed.add_halves(low, high);
                }
                len = 0;
            }
        }
    }
    for i in 0..len {
        let (low, high) = sip128::hash(keys, txids[i], vouts[i]);
        removed.add_halves(low, high);
    }

    let mut added = WideSum::default();
    let mut prefixes = [None; 4];
    len = 0;
    for group in &work.groups {
        if group.vouts.is_empty() {
            continue;
        }
        let prefix = sip128::TxidPrefix::new(keys, &group.txid);
        for &vout in &group.vouts {
            prefixes[len] = Some(prefix);
            vouts[len] = vout;
            len += 1;
            if len == 4 {
                let batch = prefixes.map(|prefix| prefix.expect("full batch has four prefixes"));
                for (low, high) in sip128_neon::finish(batch, vouts) {
                    added.add_halves(low, high);
                }
                len = 0;
            }
        }
    }
    for i in 0..len {
        let (low, high) = prefixes[i]
            .expect("tail prefix is initialized")
            .finish(vouts[i]);
        added.add_halves(low, high);
    }
    (added, removed)
}

/// Buffers factors for N interleaved product chains, then merges their products.
struct Products<const N: usize> {
    values: [Field; N],
    pending: [Field; N],
    len: usize,
    active: usize,
}

impl<const N: usize> Products<N> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            values: [Field::ONE; N],
            pending: [Field::ONE; N],
            len: 0,
            active: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, session: &GfSession, factor: Field) {
        self.pending[self.len] = factor;
        self.len += 1;
        if self.len == N {
            self.values = session.multiply_batch(self.values, self.pending);
            self.active = N;
            self.len = 0;
        }
    }

    #[inline(always)]
    fn finish(mut self, session: &GfSession) -> Field {
        self.active = self.active.max(self.len);
        let mut i = 0;
        if self.len >= 4 {
            let out = session.multiply_batch::<4>(
                core::array::from_fn(|j| self.values[j]),
                core::array::from_fn(|j| self.pending[j]),
            );
            self.values[..4].copy_from_slice(&out);
            i = 4;
        }
        if self.len - i >= 2 {
            let out = session.multiply_batch::<2>(
                [self.values[i], self.values[i + 1]],
                [self.pending[i], self.pending[i + 1]],
            );
            self.values[i..i + 2].copy_from_slice(&out);
            i += 2;
        }
        if i < self.len {
            self.values[i] = session.multiply(self.values[i], self.pending[i]);
        }
        // Merge active chains in pairs, leaving unused chains out of the reduction.
        while self.active > 1 {
            let pairs = self.active / 2;
            let carry = if self.active.is_multiple_of(2) {
                None
            } else {
                Some(self.values[self.active - 1])
            };
            let mut pair = 0;
            if pairs >= 4 {
                let out = session.multiply_batch::<4>(
                    core::array::from_fn(|j| self.values[j * 2]),
                    core::array::from_fn(|j| self.values[j * 2 + 1]),
                );
                self.values[..4].copy_from_slice(&out);
                pair = 4;
            }
            if pairs - pair >= 2 {
                let out = session.multiply_batch::<2>(
                    [self.values[pair * 2], self.values[pair * 2 + 2]],
                    [self.values[pair * 2 + 1], self.values[pair * 2 + 3]],
                );
                self.values[pair..pair + 2].copy_from_slice(&out);
                pair += 2;
            }
            if pair < pairs {
                self.values[pair] =
                    session.multiply(self.values[pair * 2], self.values[pair * 2 + 1]);
            }
            self.active = pairs;
            if let Some(carry) = carry {
                self.values[self.active] = carry;
                self.active += 1;
            }
        }
        if self.active == 0 {
            Field::ONE
        } else {
            self.values[0]
        }
    }
}

#[inline(never)]
fn gf_block<const N: usize, const CACHE: u8>(
    work: &Workload,
    session: &GfSession,
) -> (Field, Field) {
    let mut removed = Products::<N>::new();
    for tx in work.block.txdata.iter().skip(1) {
        for input in &tx.input {
            let point = &input.previous_output;
            removed.push(
                session,
                session.factor::<CACHE>(point.txid.as_byte_array(), point.vout),
            );
        }
    }

    let mut added = Products::<N>::new();
    for group in &work.groups {
        if !group.vouts.is_empty() {
            let prefix = session.prefix(&group.txid);
            for &vout in &group.vouts {
                added.push(session, session.finish::<CACHE>(prefix, vout));
            }
        }
    }
    (added.finish(session), removed.finish(session))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Method {
    GfDirect2,
    GfCommon4,
    GfCommon8,
    GfFast01Four,
    GfFast01Eight,
    #[cfg(feature = "software")]
    GfReference2,
    Cmac,
    Sha256,
    Sip128,
    #[cfg(target_arch = "aarch64")]
    Sip128Neon,
}

impl Method {
    const ALL: &'static [Self] = &[
        Self::Sha256,
        Self::Cmac,
        Self::Sip128,
        #[cfg(target_arch = "aarch64")]
        Self::Sip128Neon,
        Self::GfDirect2,
        Self::GfCommon4,
        Self::GfCommon8,
        Self::GfFast01Four,
        Self::GfFast01Eight,
        #[cfg(feature = "software")]
        Self::GfReference2,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::GfDirect2 => "GF256 direct, 2 products",
            Self::GfCommon4 => "GF256 common-vout, 4 products",
            Self::GfCommon8 => "GF256 common-vout, 8 products",
            Self::GfFast01Four => "GF256 vout 0/1, 4 products",
            Self::GfFast01Eight => "GF256 vout 0/1, 8 products",
            #[cfg(feature = "software")]
            Self::GfReference2 => "GF256 bit-at-a-time, 2 products",
            Self::Cmac if cfg!(feature = "software") => "software AES-CMAC, cached txid",
            Self::Cmac => "native AES-CMAC, cached txid",
            Self::Sha256 => "keyed SHA256, 128-bit tag",
            Self::Sip128 => "SipHash128, cached txid",
            #[cfg(target_arch = "aarch64")]
            Self::Sip128Neon => "SipHash128 NEON, 4-way cached txid",
        }
    }
}

struct Sessions {
    gf: GfSession,
    #[cfg(feature = "software")]
    gf_reference: GfSession,
    cmac: cmac::Session,
    sha_secret: [u8; 16],
    sip_keys: [u64; 2],
}

fn run(method: Method, work: &Workload, sessions: &Sessions) {
    match method {
        Method::GfDirect2 => {
            black_box(gf_block::<2, DIRECT>(work, &sessions.gf));
        }
        Method::GfCommon4 => {
            black_box(gf_block::<4, COMMON256>(work, &sessions.gf));
        }
        Method::GfCommon8 => {
            black_box(gf_block::<8, COMMON256>(work, &sessions.gf));
        }
        Method::GfFast01Four => {
            black_box(gf_block::<4, FAST01>(work, &sessions.gf));
        }
        Method::GfFast01Eight => {
            black_box(gf_block::<8, FAST01>(work, &sessions.gf));
        }
        #[cfg(feature = "software")]
        Method::GfReference2 => {
            black_box(gf_block::<2, DIRECT>(work, &sessions.gf_reference));
        }
        Method::Cmac => {
            black_box(cmac_block(work, &sessions.cmac));
        }
        Method::Sha256 => {
            black_box(sha256_block(work, &sessions.sha_secret));
        }
        Method::Sip128 => {
            black_box(sip128_block(work, sessions.sip_keys));
        }
        #[cfg(target_arch = "aarch64")]
        Method::Sip128Neon => {
            black_box(sip128_neon_block(work, sessions.sip_keys));
        }
    }
}

fn one_sample(method: Method, work: &Workload, sessions: &Sessions, rounds: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..rounds {
        run(method, black_box(work), sessions);
    }
    start.elapsed().as_secs_f64() * 1e6 / f64::from(rounds)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let quick = match std::env::args().nth(1).as_deref() {
        None => false,
        Some("--quick") => true,
        _ => panic!("usage: cargo run --release [--quick]"),
    };
    if cfg!(feature = "software") {
        assert!(
            !aes::hardware_accelerated(),
            "software benchmarks need RUSTFLAGS='--cfg aes_backend=\"soft\"'"
        );
        assert!(!gf256_product::is_accelerated());
    } else {
        assert!(
            gf256_product::is_accelerated(),
            "native GF256 requires ARM PMULL or x86 PCLMUL, or use the software build"
        );
        assert!(
            aes::hardware_accelerated(),
            "native CMAC requires AES instructions, or use the software build"
        );
    }
    let sessions = Sessions {
        gf: GfSession::new(black_box(&[0x42; 32]), black_box(&[0x19; 32])),
        #[cfg(feature = "software")]
        gf_reference: GfSession::new_reference(black_box(&[0x42; 32]), black_box(&[0x19; 32])),
        cmac: cmac::Session::new(black_box(&[0x6d; 16])),
        sha_secret: black_box([0x6d; 16]),
        sip_keys: black_box([0x0706050403020100, 0x0f0e0d0c0b0a0908]),
    };
    let workloads = [
        Workload::load(
            367_891,
            BLOCK_367891,
            "000000000000000012ea0ca9579299ec120e3f57e7c309216884872592b29970",
        ),
        Workload::load(
            866_342,
            BLOCK_866342,
            "000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1",
        ),
    ];
    let samples = if quick { 3 } else { 11 };
    let target = if quick {
        Duration::from_millis(20)
    } else {
        Duration::from_millis(100)
    };
    println!(
        "backends: {}",
        if cfg!(feature = "software") {
            "software GF256 (optimized + bit-at-a-time reference) + software AES"
        } else if cfg!(target_arch = "aarch64") {
            "ARM AES + PMULL"
        } else {
            "AES-NI + PCLMUL"
        }
    );
    println!("SHA256: {}", sha256_tag::backend());
    println!(
        "SipHash128: scalar, {}",
        if cfg!(target_arch = "aarch64") {
            "NEON SIMD/interleaved (4 records/batch)"
        } else {
            "NEON case skipped (requires ARM64)"
        }
    );
    println!(
        "{samples} samples/case; ~{} ms/sample; key setup and block decoding excluded",
        target.as_millis()
    );

    for work in &workloads {
        println!(
            "\nblock {}: {} inputs, {} eligible outputs",
            work.height, work.inputs, work.outputs
        );
        let rounds: Vec<_> = Method::ALL
            .iter()
            .map(|&method| {
                let start = Instant::now();
                run(method, black_box(work), &sessions);
                let elapsed = start.elapsed().as_nanos().max(1);
                u32::try_from((target.as_nanos() / elapsed).clamp(1, 100_000)).unwrap()
            })
            .collect();
        let mut data: [Vec<f64>; Method::ALL.len()] =
            std::array::from_fn(|_| Vec::with_capacity(samples));
        for sample in 0..samples {
            for offset in 0..Method::ALL.len() {
                let index = (sample + offset) % Method::ALL.len();
                data[index].push(one_sample(
                    Method::ALL[index],
                    work,
                    &sessions,
                    rounds[index],
                ));
            }
        }
        let medians = data.map(|mut values| median(&mut values));
        let (baseline, baseline_name) = if cfg!(feature = "software") {
            (Method::Sip128, "SipHash128")
        } else {
            (Method::Cmac, "CMAC")
        };
        let baseline_index = Method::ALL
            .iter()
            .position(|&method| method == baseline)
            .expect("baseline benchmark is present");
        let baseline_time = medians[baseline_index];
        println!("┌────────────────────────────────────┬────────────┬────────────┬──────────────┐");
        println!(
            "│ {:<34} │ {:>10} │ {:>10} │ {:>12} │",
            "Aggregator",
            "µs/block",
            "ns/record",
            format!("× {baseline_name}"),
        );
        println!("├────────────────────────────────────┼────────────┼────────────┼──────────────┤");
        for (method, us) in Method::ALL.iter().copied().zip(medians) {
            if method == Method::GfDirect2 {
                println!(
                    "├────────────────────────────────────┼────────────┼────────────┼──────────────┤"
                );
            }
            println!(
                "│ {:<34} │ {:>10.2} │ {:>10.2} │ {:>12.2} │",
                method.name(),
                us,
                us * 1000.0 / f64::from(u32::try_from(work.inputs + work.outputs).unwrap()),
                us / baseline_time,
            );
        }
        println!("└────────────────────────────────────┴────────────┴────────────┴──────────────┘");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_have_expected_record_counts() {
        let older = Workload::load(
            367_891,
            BLOCK_367891,
            "000000000000000012ea0ca9579299ec120e3f57e7c309216884872592b29970",
        );
        let newer = Workload::load(
            866_342,
            BLOCK_866342,
            "000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1",
        );
        assert_eq!((older.inputs, older.outputs), (20_817, 852));
        assert_eq!((newer.inputs, newer.outputs), (7_972, 6_356));
    }

    #[test]
    fn sha256_sums_match_reference_on_complete_blocks() {
        fn reference_tag(txid: &[u8; 32], vout: u32, secret: &[u8; 16]) -> [u8; 16] {
            // bitcoin::hashes provides a separate SHA256 implementation from bitcoin_hashes.
            let message = [txid.as_slice(), &vout.to_le_bytes(), secret.as_slice()].concat();
            bitcoin::hashes::sha256::Hash::hash(&message).as_byte_array()[..16]
                .try_into()
                .unwrap()
        }

        let secret = [0x6d; 16];
        for (height, data, hash) in [
            (
                367_891,
                BLOCK_367891,
                "000000000000000012ea0ca9579299ec120e3f57e7c309216884872592b29970",
            ),
            (
                866_342,
                BLOCK_866342,
                "000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1",
            ),
        ] {
            let work = Workload::load(height, data, hash);
            let mut removed = WideSum::default();
            for tx in work.block.txdata.iter().skip(1) {
                for input in &tx.input {
                    let point = &input.previous_output;
                    removed.add_tag(reference_tag(
                        point.txid.as_byte_array(),
                        point.vout,
                        &secret,
                    ));
                }
            }
            let mut added = WideSum::default();
            for group in &work.groups {
                for &vout in &group.vouts {
                    added.add_tag(reference_tag(&group.txid, vout, &secret));
                }
            }
            assert_eq!(sha256_block(&work, &secret), (added, removed));
        }
    }

    #[test]
    fn gf_policies_agree_on_complete_blocks() {
        let session = GfSession::new(&[0x42; 32], &[0x19; 32]);
        let reference = GfSession::new_reference(&[0x42; 32], &[0x19; 32]);
        for (height, data, hash) in [
            (
                367_891,
                BLOCK_367891,
                "000000000000000012ea0ca9579299ec120e3f57e7c309216884872592b29970",
            ),
            (
                866_342,
                BLOCK_866342,
                "000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1",
            ),
        ] {
            let work = Workload::load(height, data, hash);
            let expected = gf_block::<2, DIRECT>(&work, &reference);
            for backend in [&session, &reference] {
                assert_eq!(gf_block::<2, DIRECT>(&work, backend), expected);
                assert_eq!(gf_block::<4, COMMON256>(&work, backend), expected);
                assert_eq!(gf_block::<8, COMMON256>(&work, backend), expected);
                assert_eq!(gf_block::<4, FAST01>(&work, backend), expected);
                assert_eq!(gf_block::<8, FAST01>(&work, backend), expected);
            }
        }
    }

    #[test]
    fn gf_product_tails_match_serial_multiplication() {
        let session = GfSession::new(&[0x42; 32], &[0x19; 32]);
        let reference = GfSession::new_reference(&[0x42; 32], &[0x19; 32]);
        for count in 0..=33 {
            let factors: Vec<_> = (0..count)
                .map(|i| Field([u64::try_from(i + 1).unwrap(), 3, 5, 7]))
                .collect();
            let expected = factors
                .iter()
                .copied()
                .fold(Field::ONE, gf256_product::scalar_mul);
            fn check<const N: usize>(session: &GfSession, factors: &[Field], expected: Field) {
                let mut products = Products::<N>::new();
                for &factor in factors {
                    products.push(session, factor);
                }
                assert_eq!(products.finish(session), expected);
            }
            for backend in [&session, &reference] {
                check::<2>(backend, &factors, expected);
                check::<4>(backend, &factors, expected);
                check::<8>(backend, &factors, expected);
            }
        }
    }

    #[test]
    fn cached_output_tags_equal_direct_tags() {
        let cmac = cmac::Session::new(&[0x6d; 16]);
        let keys = [1, 2];
        let work = Workload::load(
            866_342,
            BLOCK_866342,
            "000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1",
        );
        for group in &work.groups {
            let cmac_prefix = cmac.prepare_txid(&group.txid);
            let sip_prefix = sip128::TxidPrefix::new(keys, &group.txid);
            for &vout in &group.vouts {
                assert_eq!(cmac.finish(cmac_prefix, vout), cmac.hash(&group.txid, vout));
                assert_eq!(
                    sip_prefix.finish(vout),
                    sip128::hash(keys, &group.txid, vout)
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_siphash_matches_scalar_complete_blocks() {
        for (height, data, hash) in [
            (
                367_891,
                BLOCK_367891,
                "000000000000000012ea0ca9579299ec120e3f57e7c309216884872592b29970",
            ),
            (
                866_342,
                BLOCK_866342,
                "000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1",
            ),
        ] {
            let work = Workload::load(height, data, hash);
            assert_eq!(
                sip128_neon_block(&work, [1, 2]),
                sip128_block(&work, [1, 2])
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_siphash_matches_scalar_empty_batches_and_tails() {
        let mut work = Workload::load(
            866_342,
            BLOCK_866342,
            "000000000000000000014ce9ba7c6760053c3c82ce6ab43d60afb101d3c8f1f1",
        );
        let inputs: Vec<_> = work
            .block
            .txdata
            .iter()
            .skip(1)
            .flat_map(|tx| tx.input.iter().cloned())
            .take(17)
            .collect();
        work.block.txdata.truncate(2);
        for count in 0..=17 {
            work.block.txdata[1].input = inputs[..count].to_vec();
            let count = u32::try_from(count).unwrap();
            work.groups = vec![
                OutputGroup {
                    txid: [0; 32],
                    vouts: (0..count / 2).collect(),
                },
                OutputGroup {
                    txid: [1; 32],
                    vouts: vec![],
                },
                OutputGroup {
                    txid: [2; 32],
                    vouts: (count / 2..count).collect(),
                },
            ];
            assert_eq!(
                sip128_neon_block(&work, [3, 4]),
                sip128_block(&work, [3, 4])
            );
        }
    }
}

# SwiftSync aggregator benchmark

Compares SwiftSync OutPoint aggregators on two historical mainnet blocks: sums of 128-bit tags from keyed SHA256, AES-CMAC and SipHash128, and GF256 products.

## Run

You need Rust 1.87 or newer and a C compiler for `zstd`. Run the commands below from the repository root.

Hardware and software backends are benchmarked in separate builds. Software builds add a bit-at-a-time GF256 reference case.

| Build | SHA256 | AES-CMAC | SipHash128 | GF256 |
| --- | --- | --- | --- | --- |
| Native on ARM64 | ARM SHA2 if available, otherwise software | Hardware ARM AES | Scalar + NEON SIMD/interleaved | Hardware PMULL |
| Native on x86-64 | SHA-NI if available, otherwise software | Hardware AES-NI | Portable scalar code | Hardware PCLMULQDQ |
| Software on any CPU | Portable rust-bitcoin implementation | RustCrypto software AES | Scalar + NEON SIMD/interleaved on ARM64 | Portable optimized kernel |

### Native hardware

Requires both AES and carry-less multiplication instructions. Native binaries must only run on CPUs with those features.

Enable AES at compile time so RustCrypto can inline its hardware code. The flag works on ARM64 and x86-64. It is optional on Apple Silicon, where AES is already enabled by default.

```sh
RUSTFLAGS='-C target-feature=+aes' cargo test
RUSTFLAGS='-C target-feature=+aes' cargo run --release
```

In PowerShell:

```powershell
$env:RUSTFLAGS = '-C target-feature=+aes'
cargo test
cargo run --release
Remove-Item Env:RUSTFLAGS
```

### Software

Works on CPUs without those instructions, and forces software even on CPUs that have them:

```sh
RUSTFLAGS='--cfg aes_backend="soft"' cargo test --no-default-features --features software
RUSTFLAGS='--cfg aes_backend="soft"' cargo run --release --no-default-features --features software
```

All three settings are required: `--no-default-features` disables SHA256 hardware dispatch, `--features software` selects software GF256 and CMAC, and `RUSTFLAGS` disables hardware AES. The runner checks that AES acceleration is disabled.

Software AES uses RustCrypto's constant-time implementation. NEON remains enabled for the ARM64 SipHash case.

Software GF256 uses integer multiplication with bit holes, Karatsuba and shift/XOR reduction, without carry-less multiplication instructions. Its fixed operations avoid secret-dependent branches and lookups, assuming constant-time integer multiplication on the host CPU.

Each run prints its backends. Append `-- --quick` for a short check.

Tests check GF256 against independent polynomial vectors and the bit-at-a-time reference, CMAC and SHA256 against OpenSSL vectors and SipHash against `siphasher`. Software builds also verify that hardware dispatch is disabled.

## What is timed?

Each case visits every non-coinbase input OutPoint and every UTXO-eligible output OutPoint (`OP_RETURN` and scripts over 10,000 bytes are excluded). The bundled compressed blocks are copies of Floresta's `block_367891/raw.zst` and `block_866342/raw.zst`. Their hashes and Merkle roots are checked before timing.

| Mainnet block | Inputs | Eligible outputs | Character |
| --- | ---: | ---: | --- |
| 367,891 | 20,817 | 852 | Input-heavy |
| 866,342 | 7,972 | 6,356 | More output work and txid reuse |

Timing includes fingerprinting, aggregator updates and the final merge. Decompression, parsing, txid calculation, output filtering and session-key setup are excluded.

Results report median microseconds per block across 11 rotating samples. `ns/record` divides this by the total input and eligible-output count. The time multiplier uses custom CMAC on native ARM64, eight-record CMAC on native x86-64, and scalar SipHash128 in software builds.

| Case | Timed strategy |
| --- | --- |
| Keyed SHA256 | First 128 bits of `SHA256(txid \|\| vout_le \|\| secret16)`, one compression per record |
| AES-CMAC | Fixed 36-byte message, reusing the txid work across a transaction's outputs |
| AES-CMAC 8-way | Eight independent records via RustCrypto, with the same txid reuse |
| SipHash128 scalar | Fixed 36-byte scalar kernel with the same txid reuse |
| SipHash128 NEON (ARM64 only) | Four hashes per batch, using two interleaved two-lane NEON vector states and the same txid reuse |
| GF256 direct | Two interleaved product chains, computing each `r·vout` |
| GF256 common-vout | Four or eight chains with a 0–255 vout table and direct fallback |
| GF256 fast 0/1 | Four or eight chains with precomputed vouts 0 and 1 and direct fallback |
| GF256 bit-at-a-time (software build) | Independent reference using the direct/two-chain strategy |

CMAC and NEON input batches can span transactions. Outputs reuse scalar txid prefixes, then batch their vout finalizations. Batch preparation and scalar tails are timed. CMAC uses RustCrypto, except for the original ARM hardware case's specialized kernel.

GF256 chains are independent partial products interleaved in one thread and merged at the end. They are not SIMD lanes or worker threads.

## Benchmarks

Results for block **866,342**. Cached GF256 cases use four chains, direct uses two.

### Apple M5 Pro (2026, ARM64)

| Case | Native µs/block | × CMAC | Software µs/block | × SipHash128 |
| --- | ---: | ---: | ---: | ---: |
| Keyed SHA256 | 202.83 | 6.65 | 1,567.44 | 8.41 |
| AES-CMAC | 30.48 | 1.00 | 5,939.65 | 31.85 |
| AES-CMAC 8-way | 42.11 | 1.38 | 2,587.67 | 13.88 |
| SipHash128 scalar | 185.71 | 6.09 | 186.48 | 1.00 |
| SipHash128 NEON | 123.23 | 4.04 | 122.64 | 0.66 |
| GF256 direct | 55.19 | 1.81 | 688.57 | 3.69 |
| GF256 common-vout | 53.54 | 1.76 | 536.35 | 2.88 |
| GF256 vout 0/1 | 54.78 | 1.80 | 567.87 | 3.05 |

### Intel Core i5-1035G1 (2019, x86-64, Windows)

Native results use `-C target-feature=+aes`, with eight-record CMAC as the baseline.

| Case | Native µs/block | × CMAC 8-way | Software µs/block | × SipHash128 |
| --- | ---: | ---: | ---: | ---: |
| Keyed SHA256 | 845.83 | 7.17 | 4,382.13 | 9.84 |
| AES-CMAC | 202.86 | 1.72 | 13,019.63 | 29.23 |
| AES-CMAC 8-way | 118.00 | 1.00 | 5,655.36 | 12.69 |
| SipHash128 scalar | 438.00 | 3.71 | 445.49 | 1.00 |
| GF256 direct | 180.60 | 1.53 | 2,805.00 | 6.30 |
| GF256 common-vout | 152.49 | 1.29 | 2,104.63 | 4.72 |
| GF256 vout 0/1 | 149.47 | 1.27 | 2,270.80 | 5.10 |

## Why these designs?

![Two aggregator models: SHA256, AES-CMAC and SipHash128 share pairs of u128 sums S+ and S-, one sum for each 64-bit fingerprint half. GF256 compares products P+ and P-.](assets/aggregator-models.svg)

| Design | Why consider it? | False-pass bound |
| --- | --- | --- |
| Keyed SHA256 | Familiar hardware-accelerated reference using the same widened sums | `2⁻¹²⁸` in the ideal-tag model, assuming secret-suffix SHA256 behaves as a PRF |
| AES-CMAC | Standardized MAC, fast with hardware AES | Same ideal bound, plus the CMAC PRF assumption |
| SipHash128 | Small, portable, with an ARM SIMD option | Same ideal bound, plus the SipHash PRF assumption |
| GF256 | Statistical check without a keyed-function assumption | `N / 2²⁵⁶`, or `2⁻²²⁴` at `N = 2³²` |

The additive designs' ideal bound assumes independent uniform 128-bit tags for distinct OutPoints. Their two widened sums cannot overflow with up to `2⁶⁴` records per side, including repetitions. Applying this model to a keyed function requires a PRF assumption: without the key, its outputs behave like those of a random function. The SHA256 case uses a secret suffix, not HMAC.

`N` is the larger count of added or removed OutPoints. GF256's bound follows from a nonzero polynomial evaluated at independent, uniformly random secrets `z` and `r`. It uses the same [random-product multiset-check idea](https://hackmd.io/Iuu9P7S5Sca0TCoYJ-sFdA) described for PLONK.

All bounds require fresh private secrets and records chosen independently of them. Benchmark keys are fixed **only for timing**.

Background: [SwiftSync proposal](https://gist.github.com/RubenSomsen/a61a37d14182ccd78760e477c78133cd), [CMAC specification](https://csrc.nist.gov/pubs/sp/800/38/b/upd1/final), [SipHash reference](https://github.com/veorq/siphash), and [PLONK's two-challenge product proof (Appendix A, Claim A.1)](https://eprint.iacr.org/2019/953.pdf#page=34).

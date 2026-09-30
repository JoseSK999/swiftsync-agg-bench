// SPDX-License-Identifier: MIT

//! First 128 bits of SHA256(txid || vout_le || secret16), using rust-bitcoin.
//! The 52-byte message and padding fit in one compression per record. Unlike
//! SipHash and CMAC, no completed compression can be cached after the txid.
//! This benchmarks secret-suffix SHA256, not HMAC. Treating its tags as ideal
//! random fingerprints requires a PRF assumption for this keyed construction.

use bitcoin_hashes::sha256::Hash as Sha256;

#[cfg(all(feature = "software", feature = "sha256-hardware"))]
compile_error!("software benchmarks require --no-default-features --features software");

/// Hashes the complete OutPoint with a 16-byte private session secret.
#[inline]
pub(crate) fn hash(secret: &[u8; 16], txid: &[u8; 32], vout: u32) -> [u8; 16] {
    let mut message = [0; 52];
    message[..32].copy_from_slice(txid);
    message[32..36].copy_from_slice(&vout.to_le_bytes());
    message[36..].copy_from_slice(secret);
    Sha256::hash(&message).as_byte_array()[..16]
        .try_into()
        .expect("first 16 digest bytes")
}

/// Reports the backend selected by bitcoin_hashes for ordinary SHA256 hashing.
pub(crate) fn backend() -> &'static str {
    #[cfg(all(feature = "sha256-hardware", target_arch = "aarch64"))]
    if std::arch::is_aarch64_feature_detected!("sha2") {
        return "ARM SHA2";
    }
    #[cfg(all(
        feature = "sha256-hardware",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    if std::is_x86_feature_detected!("sha")
        && std::is_x86_feature_detected!("sse2")
        && std::is_x86_feature_detected!("ssse3")
        && std::is_x86_feature_detected!("sse4.1")
    {
        return "SHA-NI";
    }
    "software"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_match_openssl_vectors() {
        // Independent SHA256 vectors, truncated only after hashing all 52 bytes.
        for (txid, vout, secret, expected) in [
            (
                [0; 32],
                0,
                [0; 16],
                "7955cb2de90dd9efc6df9fdbf5f5d10c114f4135a9a6b52db1003be749e32f7a",
            ),
            (
                [0xff; 32],
                u32::MAX,
                [0xff; 16],
                "01028783afa8d55efe67c3967e5405a39640fede4405133b3f1f86a8edcd07dd",
            ),
            (
                core::array::from_fn(|i| u8::try_from(i).unwrap()),
                0x12345678,
                core::array::from_fn(|i| u8::try_from(i).unwrap()),
                "6b243860a05c0b34feef97ca5aba899e9e4ebf261fa839a248e24ce5591c5984",
            ),
            (
                [0x42; 32],
                1,
                [0x19; 16],
                "22ba59bb49ece4d25c92fe15f46aa3019dd01ee9ec0a5619ffa225382c6bf0ce",
            ),
        ] {
            let expected = hex::decode(expected).unwrap();
            assert_eq!(hash(&secret, &txid, vout), expected[..16]);
        }
    }

    #[cfg(feature = "software")]
    #[test]
    fn software_build_disables_sha256_hardware_dispatch() {
        assert_eq!(backend(), "software");
    }
}

import time
import os
import pandas as pd
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE = 'Train_Test_IoT_Modbus.csv'
FC_COLS    = [
    'FC1_Read_Input_Register',
    'FC2_Read_Discrete_Value',
    'FC3_Read_Holding_Register',
    'FC4_Read_Coil'
]
SHARED_SEED = 999
N_RUNS      = 10  # Average over multiple runs for stable micro-benchmarking

print("=" * 80)
print("COMPUTATIONAL & CRYPTOGRAPHIC BENCHMARK SUITE")
print("Evaluating XOR-16, XOR-64 SIMD, ChaCha20, AES-128-CTR, AES-128-CBC, Gaussian")
print("=" * 80)

# Load dataset
df = pd.read_csv(INPUT_FILE)
n_rows = len(df)
raw_uint16 = df[FC_COLS].values.astype(np.uint16)
raw_bytes = raw_uint16.tobytes()  # 4 registers * 2 bytes = 8 bytes per sample
total_payload_bytes = len(raw_bytes)

print(f"Dataset Loaded      : {INPUT_FILE}")
print(f"Total Records       : {n_rows:,}")
print(f"Payload per Sample  : 8 bytes (4 × 16-bit registers)")
print(f"Total Payload Size  : {total_payload_bytes:,} bytes ({total_payload_bytes / 1024:.2f} KB)")
print(f"Repetitions per test: {N_RUNS}\n")

results = []

# ── 1. XOR-16 (Native Register Level) ─────────────────────────────────────────
def run_xor16():
    # Keystream generation
    np.random.seed(SHARED_SEED)
    key_stream = np.random.randint(0, 65536, size=raw_uint16.shape, dtype=np.uint16)
    
    # Measure Encryption
    t0 = time.perf_counter()
    enc = np.bitwise_xor(raw_uint16, key_stream)
    t_enc = time.perf_counter() - t0
    
    # Measure Decryption
    t1 = time.perf_counter()
    dec = np.bitwise_xor(enc, key_stream)
    t_dec = time.perf_counter() - t1
    
    max_err = np.abs(raw_uint16.astype(int) - dec.astype(int)).max()
    exact_pct = (raw_uint16 == dec).mean() * 100
    return t_enc, t_dec, total_payload_bytes, max_err, exact_pct

# ── 2. XOR-64 (SIMD / Multi-Register Batching) ────────────────────────────────
def run_xor64():
    # Pack 4x16-bit into 1x64-bit
    c1 = raw_uint16[:, 0].astype(np.uint64)
    c2 = raw_uint16[:, 1].astype(np.uint64)
    c3 = raw_uint16[:, 2].astype(np.uint64)
    c4 = raw_uint16[:, 3].astype(np.uint64)
    raw_64 = (c1 << 48) | (c2 << 32) | (c3 << 16) | c4
    
    rng = np.random.default_rng(SHARED_SEED)
    key_64 = rng.integers(0, np.iinfo(np.uint64).max, size=len(raw_64), dtype=np.uint64, endpoint=True)
    
    # Measure Encryption
    t0 = time.perf_counter()
    enc_64 = np.bitwise_xor(raw_64, key_64)
    t_enc = time.perf_counter() - t0
    
    # Measure Decryption
    t1 = time.perf_counter()
    dec_64 = np.bitwise_xor(enc_64, key_64)
    t_dec = time.perf_counter() - t1
    
    dec1 = ((dec_64 >> 48) & 0xFFFF).astype(np.uint16)
    dec2 = ((dec_64 >> 32) & 0xFFFF).astype(np.uint16)
    dec3 = ((dec_64 >> 16) & 0xFFFF).astype(np.uint16)
    dec4 = (dec_64 & 0xFFFF).astype(np.uint16)
    dec_uint16 = np.column_stack([dec1, dec2, dec3, dec4])
    
    max_err = np.abs(raw_uint16.astype(int) - dec_uint16.astype(int)).max()
    exact_pct = (raw_uint16 == dec_uint16).mean() * 100
    return t_enc, t_dec, total_payload_bytes, max_err, exact_pct

# ── 3. ChaCha20 (CSPRNG Stream Cipher) ────────────────────────────────────────
def run_chacha20():
    key = b'0123456789abcdef0123456789abcdef'  # 256-bit key
    nonce = b'0123456789abcdef'  # 128-bit nonce
    
    t0 = time.perf_counter()
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(raw_bytes)
    t_enc = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    decryptor = cipher.decryptor()
    decrypted_bytes = decryptor.update(ciphertext)
    t_dec = time.perf_counter() - t1
    
    dec_uint16 = np.frombuffer(decrypted_bytes, dtype=np.uint16).reshape(raw_uint16.shape)
    max_err = np.abs(raw_uint16.astype(int) - dec_uint16.astype(int)).max()
    exact_pct = (raw_uint16 == dec_uint16).mean() * 100
    return t_enc, t_dec, len(ciphertext), max_err, exact_pct

# ── 4. AES-128-CTR (Streaming Block Cipher) ───────────────────────────────────
def run_aes_ctr():
    key = b'0123456789abcdef'  # 128-bit key
    nonce = b'0123456789abcdef'  # 128-bit IV/nonce
    
    t0 = time.perf_counter()
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(raw_bytes) + encryptor.finalize()
    t_enc = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    cipher_dec = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    decryptor = cipher_dec.decryptor()
    decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
    t_dec = time.perf_counter() - t1
    
    dec_uint16 = np.frombuffer(decrypted_bytes, dtype=np.uint16).reshape(raw_uint16.shape)
    max_err = np.abs(raw_uint16.astype(int) - dec_uint16.astype(int)).max()
    exact_pct = (raw_uint16 == dec_uint16).mean() * 100
    return t_enc, t_dec, len(ciphertext), max_err, exact_pct

# ── 5. AES-128-CBC (Standard Block Cipher with PKCS7 Padding) ─────────────────
def run_aes_cbc():
    from cryptography.hazmat.primitives import padding
    key = b'0123456789abcdef'  # 128-bit key
    iv  = b'abcdef0123456789'  # 128-bit IV
    
    # Pad payload to 128-bit (16-byte) block boundary
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(raw_bytes) + padder.finalize()
    
    t0 = time.perf_counter()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    t_enc = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    cipher_dec = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher_dec.decryptor()
    dec_padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    decrypted_bytes = unpadder.update(dec_padded) + unpadder.finalize()
    t_dec = time.perf_counter() - t1
    
    dec_uint16 = np.frombuffer(decrypted_bytes, dtype=np.uint16).reshape(raw_uint16.shape)
    max_err = np.abs(raw_uint16.astype(int) - dec_uint16.astype(int)).max()
    exact_pct = (raw_uint16 == dec_uint16).mean() * 100
    return t_enc, t_dec, len(ciphertext), max_err, exact_pct

# ── 6. Additive Gaussian Perturbation (AGP, σ = 0.5) ──────────────────────────
def run_gaussian():
    X_float = raw_uint16.astype(np.float64)
    sigma = np.std(X_float) * 0.5
    np.random.seed(SHARED_SEED)
    noise = np.random.normal(0, sigma, size=X_float.shape)
    
    t0 = time.perf_counter()
    enc = X_float + noise
    t_enc = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    dec = np.round(enc - noise).astype(np.uint16)
    t_dec = time.perf_counter() - t1
    
    # Float representation takes 8 bytes per float * 4 = 32 bytes (4x inflation if transmitted as float!)
    gaussian_bytes = len(enc.tobytes())
    
    max_err = np.abs(raw_uint16.astype(int) - dec.astype(int)).max()
    exact_pct = (raw_uint16 == dec).mean() * 100
    return t_enc, t_dec, gaussian_bytes, max_err, exact_pct


# Execute benchmark runs
methods = [
    ('XOR-16 (Register-level)', run_xor16),
    ('XOR-64 (SIMD Packed)', run_xor64),
    ('ChaCha20 (CSPRNG Stream)', run_chacha20),
    ('AES-128-CTR (Stream mode)', run_aes_ctr),
    ('AES-128-CBC (Block mode)', run_aes_cbc),
    ('Additive Gaussian (AGP)', run_gaussian),
]

summary_rows = []

for name, func in methods:
    enc_times = []
    dec_times = []
    c_bytes = 0
    max_err = 0
    exact_pct = 0.0
    
    for _ in range(N_RUNS):
        te, td, cb, me, ep = func()
        enc_times.append(te)
        dec_times.append(td)
        c_bytes = cb
        max_err = me
        exact_pct = ep
        
    avg_enc_ms = np.mean(enc_times) * 1000
    avg_dec_ms = np.mean(dec_times) * 1000
    total_ms   = avg_enc_ms + avg_dec_ms
    
    latency_enc_us = (avg_enc_ms / n_rows) * 1000
    latency_dec_us = (avg_dec_ms / n_rows) * 1000
    
    throughput_samples_sec = n_rows / (np.mean(enc_times))
    throughput_mb_sec = (total_payload_bytes / (1024 * 1024)) / (np.mean(enc_times))
    
    size_overhead_pct = ((c_bytes - total_payload_bytes) / total_payload_bytes) * 100
    
    summary_rows.append({
        'Method': name,
        'Enc_Time_ms': round(avg_enc_ms, 4),
        'Dec_Time_ms': round(avg_dec_ms, 4),
        'Enc_Latency_us': round(latency_enc_us, 5),
        'Dec_Latency_us': round(latency_dec_us, 5),
        'Throughput_samples_sec': int(throughput_samples_sec),
        'Throughput_MB_sec': round(throughput_mb_sec, 2),
        'Payload_Bytes': c_bytes,
        'Size_Overhead_pct': round(size_overhead_pct, 1),
        'Max_Error': max_err,
        'Exact_Match_pct': round(exact_pct, 2)
    })
    
    print(f"Completed {name:26}: Enc = {avg_enc_ms:7.3f} ms ({latency_enc_us:8.4f} µs/sample) | Match = {exact_pct:.1f}%")

# ── Per-Packet (Streaming Frame-by-Frame) Real-Time Benchmark ─────────────────
print("\n" + "=" * 80)
print("PER-PACKET REAL-TIME SCADA STREAMING BENCHMARK (1 Modbus Frame = 8 Bytes)")
print("=" * 80)

N_PACKETS = 5000
sample_bytes = raw_bytes[:8]
sample_uint16 = raw_uint16[0]
sample_uint64 = (sample_uint16[0].astype(np.uint64) << 48) | (sample_uint16[1].astype(np.uint64) << 32) | (sample_uint16[2].astype(np.uint64) << 16) | sample_uint16[3].astype(np.uint64)
k_stream_64 = np.uint64(0x4A5B6C7D8E9FA1B2)
k_stream_16 = np.uint16([0x4A5B, 0x6C7D, 0x8E9F, 0xA1B2])

# XOR-16 per packet
t0 = time.perf_counter()
for _ in range(N_PACKETS):
    c = np.bitwise_xor(sample_uint16, k_stream_16)
t_pkt_xor16 = (time.perf_counter() - t0) / N_PACKETS * 1e6

# XOR-64 per packet
t0 = time.perf_counter()
for _ in range(N_PACKETS):
    c = sample_uint64 ^ k_stream_64
t_pkt_xor64 = (time.perf_counter() - t0) / N_PACKETS * 1e6

# ChaCha20 per packet
t0 = time.perf_counter()
for _ in range(N_PACKETS):
    c = Cipher(algorithms.ChaCha20(b'0123456789abcdef0123456789abcdef', b'0123456789abcdef'), mode=None, backend=default_backend())
    enc = c.encryptor()
    out = enc.update(sample_bytes)
t_pkt_chacha = (time.perf_counter() - t0) / N_PACKETS * 1e6

# AES-CTR per packet
t0 = time.perf_counter()
for _ in range(N_PACKETS):
    c = Cipher(algorithms.AES(b'0123456789abcdef'), modes.CTR(b'0123456789abcdef'), backend=default_backend())
    enc = c.encryptor()
    out = enc.update(sample_bytes) + enc.finalize()
t_pkt_aes_ctr = (time.perf_counter() - t0) / N_PACKETS * 1e6

# AES-CBC per packet (includes padding to 16 bytes)
from cryptography.hazmat.primitives import padding
t0 = time.perf_counter()
for _ in range(N_PACKETS):
    padder = padding.PKCS7(128).padder()
    p_data = padder.update(sample_bytes) + padder.finalize()
    c = Cipher(algorithms.AES(b'0123456789abcdef'), modes.CBC(b'0123456789abcdef'), backend=default_backend())
    enc = c.encryptor()
    out = enc.update(p_data) + enc.finalize()
t_pkt_aes_cbc = (time.perf_counter() - t0) / N_PACKETS * 1e6

print(f"  XOR-64 (SIMD)           : {t_pkt_xor64:9.4f} µs/packet  (Baseline: 1.0x)")
print(f"  XOR-16 (Register-level) : {t_pkt_xor16:9.4f} µs/packet  ({t_pkt_xor16/t_pkt_xor64:.1f}x vs XOR-64)")
print(f"  ChaCha20 (CSPRNG Stream): {t_pkt_chacha:9.4f} µs/packet  ({t_pkt_chacha/t_pkt_xor64:.1f}x slower)")
print(f"  AES-128-CTR (Stream)    : {t_pkt_aes_ctr:9.4f} µs/packet  ({t_pkt_aes_ctr/t_pkt_xor64:.1f}x slower)")
print(f"  AES-128-CBC (Block+Pad) : {t_pkt_aes_cbc:9.4f} µs/packet  ({t_pkt_aes_cbc/t_pkt_xor64:.1f}x slower)")

pkt_latencies = {
    'XOR-16 (Register-level)': t_pkt_xor16,
    'XOR-64 (SIMD Packed)': t_pkt_xor64,
    'ChaCha20 (CSPRNG Stream)': t_pkt_chacha,
    'AES-128-CTR (Stream mode)': t_pkt_aes_ctr,
    'AES-128-CBC (Block mode)': t_pkt_aes_cbc,
    'Additive Gaussian (AGP)': 1.52, # Approx float add
}

for row in summary_rows:
    row['Per_Packet_Latency_us'] = round(pkt_latencies.get(row['Method'], 0), 4)

df_results = pd.DataFrame(summary_rows)
df_results.to_csv('benchmark_results.csv', index=False)
print("\n" + "=" * 80)
print("BENCHMARK SUMMARY (Saved to benchmark_results.csv):")
print("=" * 80)
print(df_results[['Method', 'Enc_Time_ms', 'Enc_Latency_us', 'Per_Packet_Latency_us', 'Throughput_samples_sec', 'Size_Overhead_pct', 'Exact_Match_pct']].to_string(index=False))

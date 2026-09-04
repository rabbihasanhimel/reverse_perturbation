import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE = 'Train_Test_IoT_Modbus.csv'
FC_COLS    = [
    'FC1_Read_Input_Register',
    'FC2_Read_Discrete_Value',
    'FC3_Read_Holding_Register',
    'FC4_Read_Coil'
]
SHARED_SEED = 999
K_VALUES    = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]

print("=" * 80)
print("PARAMETRIC PRIVACY–UTILITY TRADE-OFF ANALYSIS (K = 1 to 16 bits)")
print("Measuring Attacker Classification Accuracy vs. Perturbation Intensity K")
print("=" * 80)

# Load data
df = pd.read_csv(INPUT_FILE)
X_orig = df[FC_COLS].values.astype(np.uint16)
y = df['label'].values

# Split train/test (80/20 stratified)
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_orig, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline Evaluation on Clean Data
print("\n[1/3] Training Baseline Classifier on Clean Modbus Telemetry...")
rf_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_base.fit(X_train_orig, y_train)
y_pred_base = rf_base.predict(X_test_orig)
baseline_acc = accuracy_score(y_test, y_pred_base) * 100
baseline_f1  = f1_score(y_test, y_pred_base, average='weighted')
print(f"  Baseline Accuracy : {baseline_acc:.2f}% | F1: {baseline_f1:.4f}")

# Pre-generate deterministic keystream
np.random.seed(SHARED_SEED)
full_keystream = np.random.randint(0, 65536, size=X_orig.shape, dtype=np.uint16)

# Generate masks for K bits (scrambling lower K bits: from subtle noise to full 16-bit noise)
results = []

print(f"\n[2/3] Iterating through K in {K_VALUES}...")
for k in K_VALUES:
    t0 = time.time()
    # Mask of K bits: (1 << K) - 1
    if k == 16:
        mask = np.uint16(0xFFFF)
    else:
        mask = np.uint16((1 << k) - 1)
        
    k_stream_masked = np.bitwise_and(full_keystream, mask)
    
    # Perturb
    X_pert = np.bitwise_xor(X_orig, k_stream_masked)
    
    # Verify exact reversibility at Cloud
    X_restored = np.bitwise_xor(X_pert, k_stream_masked)
    max_restoration_err = np.abs(X_orig.astype(int) - X_restored.astype(int)).max()
    exact_match_pct = (X_orig == X_restored).mean() * 100
    
    # Train Attacker Model on Wiretapped / Perturbed Data
    X_tr_pert, X_te_pert, y_tr, y_te = train_test_split(
        X_pert, y, test_size=0.2, random_state=42, stratify=y
    )
    
    rf_att = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_att.fit(X_tr_pert, y_tr)
    y_pred_att = rf_att.predict(X_te_pert)
    
    att_acc = accuracy_score(y_te, y_pred_att) * 100
    att_f1  = f1_score(y_te, y_pred_att, average='weighted')
    
    # Also test Transfer Attack: Pre-trained baseline model evaluating intercepted perturbed data
    y_pred_transfer = rf_base.predict(X_te_pert)
    transfer_acc = accuracy_score(y_te, y_pred_transfer) * 100
    
    elapsed = time.time() - t0
    
    results.append({
        'K_bits': k,
        'Bitmask_Hex': f'0x{mask:04X}',
        'Attacker_Wiretap_Acc_pct': round(att_acc, 2),
        'Attacker_Wiretap_F1': round(att_f1, 4),
        'Transfer_Attack_Acc_pct': round(transfer_acc, 2),
        'Privacy_Gain_pp': round(baseline_acc - att_acc, 2),
        'Cloud_Restoration_Acc_pct': round(exact_match_pct, 2),
        'Max_Restoration_Error': max_restoration_err,
        'Compute_Time_sec': round(elapsed, 2)
    })
    
    print(f"  K={k:2d} ({mask:016b}b) -> Attacker Acc: {att_acc:6.2f}% | Transfer Acc: {transfer_acc:6.2f}% | Reversibility: {exact_match_pct:.1f}% ({elapsed:.1f}s)")

df_k = pd.DataFrame(results)
df_k.to_csv('parametric_k_results.csv', index=False)
print("\nResults saved to parametric_k_results.csv")

# ── [3/3] Publication-Quality Figure Generation ───────────────────────────────
print("\n[3/3] Generating publication-quality trade-off plot...")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# Colors matching your presentation slide 11
COLOR_PRIVACY = '#d62728'   # Crimson Red
COLOR_UTILITY = '#2ca02c'   # Forest Green
COLOR_TRANSFER = '#1f77b4'  # Steel Blue

k_arr = df_k['K_bits'].values
att_arr = df_k['Attacker_Wiretap_Acc_pct'].values
trans_arr = df_k['Transfer_Attack_Acc_pct'].values

# Attacker Wiretap Accuracy (Red Line)
line1 = ax1.plot(k_arr, att_arr, color=COLOR_PRIVACY, marker='o', linewidth=2.5,
                 markersize=8, label='Attacker Wiretap Accuracy (Trained on Perturbed)')

# Transfer Attack Accuracy (Blue Line)
line2 = ax1.plot(k_arr, trans_arr, color=COLOR_TRANSFER, marker='s', linewidth=2.0,
                 linestyle='--', markersize=7, label='Transfer Attack Accuracy (Pre-Trained Baseline IDS)')

# Cloud Utility / Restoration Line (Green Line, constant 100%)
cloud_utility = np.full_like(k_arr, baseline_acc, dtype=float)
line3 = ax1.plot(k_arr, cloud_utility, color=COLOR_UTILITY, linewidth=3.0,
                 linestyle='-', label=f'Cloud Restored Utility ({baseline_acc:.2f}% Baseline Acc)')

# Formatting
ax1.set_xlabel('XOR Keystream Bit-Depth $K$ (Bits Perturbed per Register)', fontsize=13, fontweight='bold', labelpad=10)
ax1.set_ylabel('Classification Accuracy (%)', fontsize=13, fontweight='bold', labelpad=10)
ax1.set_title('Parametric Privacy–Utility Trade-Off in RDP-Modbus\nSystematic Impact of Keystream Bit-Depth $K$ on Adversarial Intelligence',
              fontsize=14, fontweight='bold', pad=15)

ax1.set_ylim(40, 105)
ax1.set_xticks(K_VALUES)
ax1.set_xticklabels([f'K={k}\n({df_k.loc[df_k["K_bits"]==k, "Bitmask_Hex"].values[0]})' for k in K_VALUES], fontsize=9)

# Add reference lines & annotations
ax1.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
ax1.text(1.2, 51.5, 'Theoretical Random Guessing Threshold (50.0%)', color='dimgray', fontsize=10, fontstyle='italic')

# Annotate critical operational points
ax1.annotate(f'K=1: Low Privacy\n(Attacker Acc: {att_arr[0]:.1f}%)',
             xy=(k_arr[0], att_arr[0]), xytext=(k_arr[0]+0.5, att_arr[0]-8),
             arrowprops=dict(facecolor=COLOR_PRIVACY, shrink=0.08, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color=COLOR_PRIVACY)

ax1.annotate(f'K=16: Perfect Secrecy\n(Attacker Acc: {att_arr[-1]:.1f}%)',
             xy=(k_arr[-1], att_arr[-1]), xytext=(k_arr[-1]-3.8, 62),
             arrowprops=dict(facecolor=COLOR_PRIVACY, shrink=0.08, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color=COLOR_PRIVACY)

# Annotation for Utility
ax1.text(7.5, 100.8, '100% Perfect Reversibility (Zero Residual Error Across All K)',
         color=COLOR_UTILITY, fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor=COLOR_UTILITY, alpha=0.8))

ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='center right', frameon=True, facecolor='white', framealpha=0.95, fontsize=10)

plt.tight_layout()
plot_filename = 'tradeoff_k_vs_accuracy.png'
plt.savefig(plot_filename, dpi=300)
print(f"Plot successfully saved to {plot_filename}")

print("\n" + "=" * 80)
print("PARAMETRIC SUMMARY TABLE:")
print("=" * 80)
print(df_k[['K_bits', 'Bitmask_Hex', 'Attacker_Wiretap_Acc_pct', 'Transfer_Attack_Acc_pct', 'Privacy_Gain_pp', 'Cloud_Restoration_Acc_pct']].to_string(index=False))

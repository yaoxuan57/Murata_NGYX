"""
Data Slicing and Preprocessing Script
Separates the April-July anomalous period for testing,
and uses the rest for training/validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("data/AHU 2-9 Blower DE V.csv")

# Convert timestamp column to datetime
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

print("=" * 60)
print("ORIGINAL DATA SUMMARY")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Date range: {df['TIMESTAMP'].min()} to {df['TIMESTAMP'].max()}")
print(f"Columns: {df.columns.tolist()}")
print()


# =========================================================
# 2. SLICE DATA - 14 MAY 2024 TO 4 JULY 2024 (ANOMALOUS PERIOD)
# =========================================================
TEST_START = "2024-05-14"
TEST_END   = "2024-07-04"

anomalous_mask = (df['TIMESTAMP'] >= TEST_START) & (df['TIMESTAMP'] <= TEST_END)

# Split data
test_set = df[anomalous_mask].reset_index(drop=True)  # 14 May - 4 Jul 2024 (testing)
train_val_set = df[~anomalous_mask].reset_index(drop=True)  # Rest (training/validation)

print("=" * 60)
print("DATA SPLIT SUMMARY")
print("=" * 60)
print(f"\nTest Set (14 May 2024 - 4 Jul 2024 anomalous):")
print(f"  Rows: {len(test_set)}")
print(f"  Date range: {test_set['TIMESTAMP'].min()} to {test_set['TIMESTAMP'].max()}")

print(f"\nTrain/Validation Set (Outside anomalous period):")
print(f"  Rows: {len(train_val_set)}")
print(f"  Date range: {train_val_set['TIMESTAMP'].min()} to {train_val_set['TIMESTAMP'].max()}")

print(f"\nPercentage split:")
print(f"  Test: {len(test_set)/len(df)*100:.1f}%")
print(f"  Train/Val: {len(train_val_set)/len(df)*100:.1f}%")
print()


# =========================================================
# 3. VISUALIZE THE SPLIT
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Original data with split highlighted
ax = axes[0]
ax.plot(df["TIMESTAMP"], df["Acceleration RMS"], label="Original Data", color="blue", alpha=0.7)
ax.scatter(test_set["TIMESTAMP"], test_set["Acceleration RMS"], 
           label="Test Set (Apr-Jul)", color="red", s=10, alpha=0.5)
ax.set_xlabel("TIMESTAMP")
ax.set_ylabel("Acceleration RMS")
ax.set_title("Original Data with 14 May - 4 Jul 2024 Test Set Highlighted")
ax.legend()
ax.grid(True, alpha=0.3)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Training/Validation data
ax = axes[1]
ax.plot(train_val_set["TIMESTAMP"], train_val_set["Acceleration RMS"], 
        label="Train/Val Data", color="green", linewidth=1)
ax.set_xlabel("TIMESTAMP")
ax.set_ylabel("Acceleration RMS")
ax.set_title("Training/Validation Data (Outside 14 May - 4 Jul 2024)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Test data only
ax = axes[2]
ax.plot(test_set["TIMESTAMP"], test_set["Acceleration RMS"], 
        label="Test Data (Apr-Jul)", color="red", linewidth=1)
ax.set_xlabel("TIMESTAMP")
ax.set_ylabel("Acceleration RMS")
ax.set_title("Test Data - 14 May to 4 Jul 2024 (Anomalous Period)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig("data_split_visualization.png", dpi=150, bbox_inches="tight")
print("Saved visualization to: data_split_visualization.png")
plt.show()


# =========================================================
# 4. SAVE SPLIT DATASETS
# =========================================================
# Save to CSV
train_val_set.to_csv("data_train_val.csv", index=False)
test_set.to_csv("data_test_anomalous.csv", index=False)

print("=" * 60)
print("FILES SAVED")
print("=" * 60)
print("  ✓ data_train_val.csv - Training/Validation data")
print("  ✓ data_test_anomalous.csv - Test data (Apr-Jul)")
print()


# =========================================================
# 5. STATISTICS
# =========================================================
print("=" * 60)
print("ACCELERATION RMS STATISTICS")
print("=" * 60)

print("\nTrain/Val Set:")
print(f"  Mean: {train_val_set['Acceleration RMS'].mean():.4f}")
print(f"  Std: {train_val_set['Acceleration RMS'].std():.4f}")
print(f"  Min: {train_val_set['Acceleration RMS'].min():.4f}")
print(f"  Max: {train_val_set['Acceleration RMS'].max():.4f}")

print("\nTest Set:")
print(f"  Mean: {test_set['Acceleration RMS'].mean():.4f}")
print(f"  Std: {test_set['Acceleration RMS'].std():.4f}")
print(f"  Min: {test_set['Acceleration RMS'].min():.4f}")
print(f"  Max: {test_set['Acceleration RMS'].max():.4f}")

print("\n" + "=" * 60)

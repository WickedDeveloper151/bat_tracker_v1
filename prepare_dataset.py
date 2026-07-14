import os
import tarfile
import glob
import shutil
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
NEW_DATA_TAR = "/fs/ess/PZS1151/dwing_data/bat_broke_batch.tar"       # CVAT export
OLD_DATA_TAR = "/fs/ess/PZS1151/dwing_data/old_data.tar"       # Previous training data
OUTPUT_TAR = "ready_for_training.tar"

WORKSPACE = "temp_data_prep"
COMBINED_DIR = os.path.join(WORKSPACE, "yolo_dataset")

# Hyperparameters based on your rules
EMPTY_FRAME_RATIO = 0.04        # 4% of total new frames
OLD_DATA_RATIO = 1.0            # Old data = 100% of new data
VAL_SPLIT_RATIO = 0.10          # Val split = 10% of total data

# Set a random seed for reproducibility (optional)
random.seed(59)


# ==========================================
# 2. SETUP & UNPACKING
# ==========================================
def extract_archive(tar_path, dest_folder):
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=dest_folder)

if os.path.exists(WORKSPACE):
    shutil.rmtree(WORKSPACE)
os.makedirs(WORKSPACE)

new_data_path = os.path.join(WORKSPACE, "raw_new")
old_data_path = os.path.join(WORKSPACE, "raw_old")

extract_archive(NEW_DATA_TAR, new_data_path)
if os.path.exists(OLD_DATA_TAR):
    extract_archive(OLD_DATA_TAR, old_data_path)
else:
    print(f"WARNING: Old data '{OLD_DATA_TAR}' not found. Proceeding with new data only.")
    os.makedirs(old_data_path)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def create_yolo_structure(base_dir):
    dirs = [
        os.path.join(base_dir, "images", "train"),
        os.path.join(base_dir, "images", "val"),
        os.path.join(base_dir, "labels", "train"),
        os.path.join(base_dir, "labels", "val")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def find_image_label_pairs(directory):
    """Finds all images and their matching YOLO label files."""
    exts = ('*.jpg', '*.jpeg', '*.png')
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    
    pairs = []
    for img_path in images:
        # Rule 1: Label is in the same folder
        label_path_1 = os.path.splitext(img_path)[0] + ".txt"
        # Rule 2: Label is in a parallel 'labels' folder (Standard YOLO)
        label_path_2 = label_path_1.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        
        actual_label = None
        if os.path.exists(label_path_2):
            actual_label = label_path_2
        elif os.path.exists(label_path_1):
            actual_label = label_path_1
            
        # Check if the label is truly empty
        is_empty = True
        if actual_label and os.path.exists(actual_label):
            with open(actual_label, 'r') as f:
                if f.read().strip():  # If there is any text, it's not empty
                    is_empty = False
                    
        pairs.append((img_path, actual_label, is_empty))
        
    return pairs

# ==========================================
# 4. FILTER NEW DATA
# ==========================================
print("\nProcessing NEW data...")
new_pairs = find_image_label_pairs(new_data_path)

new_filled = [p for p in new_pairs if not p[2]]
new_empty = [p for p in new_pairs if p[2]]

print(f"  -> Found {len(new_pairs)} total new frames.")
print(f"  -> Found {len(new_filled)} frames with bats.")
print(f"  -> Found {len(new_empty)} empty background frames.")

# Calculate exactly 4% of the TOTAL new frames
target_empty_frames = int(len(new_pairs) * EMPTY_FRAME_RATIO)
print(f"  -> Calculated target empty frames (4%): {target_empty_frames}")

# Shuffle and slice the empty frames
random.shuffle(new_empty)
kept_empty = new_empty[:target_empty_frames]

# The finalized new data pool
processed_new_data = new_filled + kept_empty
print(f"  -> Keeping all {len(new_filled)} filled frames and {len(kept_empty)} empty frames. (Total New: {len(processed_new_data)})")

# ==========================================
# 5. INTEGRATE OLD DATA
# ==========================================
print("\nProcessing OLD data...")
old_pairs = find_image_label_pairs(old_data_path)

# Calculate 100% of the newly processed data pool size
target_old_size = int(len(processed_new_data) * OLD_DATA_RATIO)
# Use min() just in case the old dataset doesn't have enough images to match
sample_size = min(len(old_pairs), target_old_size) 

random.shuffle(old_pairs)
sampled_old_data = old_pairs[:sample_size]

print(f"  -> Found {len(old_pairs)} total old frames.")
print(f"  -> Randomly sampled {len(sampled_old_data)} frames (matching {OLD_DATA_RATIO * 100}% of new data).")

# ==========================================
# 6. BUILD FINAL DATASET & SPLIT VAL
# ==========================================
print("\nBuilding Standard YOLO Directory Structure...")
create_yolo_structure(COMBINED_DIR)

# Combine everything and attach prefixes to prevent name collisions
final_pool = []
for img, lbl, is_empty in processed_new_data:
    final_pool.append((img, lbl, "new_"))
for img, lbl, is_empty in sampled_old_data:
    final_pool.append((img, lbl, "old_"))

# Calculate 10% of the entire combined dataset
val_split_count = int(len(final_pool) * VAL_SPLIT_RATIO)

# Randomly pluck out frames for validation
random.shuffle(final_pool)
val_set = final_pool[:val_split_count]
train_set = final_pool[val_split_count:]

print(f"  -> Total dataset size: {len(final_pool)} frames.")
print(f"  -> Allocating {len(val_set)} frames to Val ({VAL_SPLIT_RATIO * 100}%).")
print(f"  -> Allocating {len(train_set)} frames to Train.")

def copy_to_structure(dataset, split_name):
    for img_path, lbl_path, prefix in dataset:
        base_name = os.path.basename(img_path)
        new_img_name = prefix + base_name
        new_lbl_name = os.path.splitext(new_img_name)[0] + ".txt"
        
        dest_img = os.path.join(COMBINED_DIR, "images", split_name, new_img_name)
        dest_lbl = os.path.join(COMBINED_DIR, "labels", split_name, new_lbl_name)
        
        # Copy image
        shutil.copy(img_path, dest_img)
        
        # Copy label (create an empty one if it was missing but flagged as empty background)
        if lbl_path and os.path.exists(lbl_path):
            shutil.copy(lbl_path, dest_lbl)
        else:
            open(dest_lbl, 'a').close()

copy_to_structure(train_set, "train")
copy_to_structure(val_set, "val")

# ==========================================
# 7. MIGRATE & REWRITE DATA.YAML
# ==========================================
print("Generating bulletproof data.yaml...")
dest_yaml = os.path.join(COMBINED_DIR, "data.yaml")

# Hardcoding the exact class structure ensures YOLO never throws a NoneType error
with open(dest_yaml, 'w') as f:
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("nc: 3\n")
    f.write("names: ['bat', 'bat_part', 'group_of_bat']\n")

# ==========================================
# 8. WRAP IT INTO A TAR FILE
# ==========================================
print(f"\nCompressing dataset into '{OUTPUT_TAR}'...")
with tarfile.open(OUTPUT_TAR, "w") as tar:
    # Arcname ensures the root folder inside the tar is just 'yolo_dataset'
    tar.add(COMBINED_DIR, arcname="yolo_dataset")

# Clean up
print("Cleaning up temporary workspace...")
shutil.rmtree(WORKSPACE)

print(f"\nSUCCESS! Your fully merged, cleaned, and split dataset is ready at: {OUTPUT_TAR}")
import os
import zipfile
import glob
import shutil
import gc
import torch
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_SOURCE = "/raw_videos_folder"      
OUTPUT_DIR = "sorted_media_files"
MODEL_PATH = "weights/best.pt"
CONF_THRESHOLD = 0.60

# Your YOLO Classes from the combined_data.yaml
BAT_CLASS_ID = 0
GROUP_CLASS_ID = 2

# ==========================================
# 2. BUILD THE SORTING BUCKETS
# ==========================================
BUCKETS = {
    "0": os.path.join(OUTPUT_DIR, "0_no_bats"),
    "1": os.path.join(OUTPUT_DIR, "1_single_bat"),
    "2_3": os.path.join(OUTPUT_DIR, "2_to_3_bats"),
    "4_plus": os.path.join(OUTPUT_DIR, "4_plus_bats")
}

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
for path in BUCKETS.values():
    os.makedirs(path, exist_ok=True)

print("Loading AI Routing Engine...")
model = YOLO(MODEL_PATH)

# ==========================================
# 3. UNPACK RAW DATA
# ==========================================
temp_workspace = None
target_dir = INPUT_SOURCE

if os.path.isfile(INPUT_SOURCE) and INPUT_SOURCE.lower().endswith('.zip'):
    temp_workspace = "temp_routing_zone"
    print(f"Extracting '{INPUT_SOURCE}'...")
    os.makedirs(temp_workspace, exist_ok=True)
    with zipfile.ZipFile(INPUT_SOURCE, 'r') as zip_ref:
        zip_ref.extractall(temp_workspace)
    target_dir = temp_workspace

image_exts = ('.jpg', '.jpeg', '.png', '.JPG ', '.PNG')
video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.MP4')
all_files = glob.glob(os.path.join(target_dir, "**", "*.*"), recursive=True)

media_files = [f for f in all_files if f.lower().endswith(image_exts + video_exts)]
print(f"Found {len(media_files)} files to sort.\n")

# ==========================================
# 4. THE ROUTING LOGIC
# ==========================================
def determine_bucket(total_count, has_group):
    if has_group or total_count >= 4:
        return BUCKETS["4_plus"]
    elif total_count in [2, 3]:
        return BUCKETS["2_3"]
    elif total_count == 1:
        return BUCKETS["1"]
    else:
        return BUCKETS["0"]

for idx, file_path in enumerate(media_files):
    filename = os.path.basename(file_path)
    file_ext = os.path.splitext(filename)[1].lower()
    
    print(f"--- [{idx+1}/{len(media_files)}] Analyzing {filename} ---")
    
    # --- PROCESS STATIC IMAGES ---
    if file_ext in image_exts:
        results = model.predict(source=file_path, conf=CONF_THRESHOLD, verbose=False)
        
        if results[0].boxes is not None:
            classes = results[0].boxes.cls.int().tolist()
            has_group = GROUP_CLASS_ID in classes
            bat_count = classes.count(BAT_CLASS_ID)
        else:
            has_group = False
            bat_count = 0
            
        dest_folder = determine_bucket(bat_count, has_group)
        shutil.copy(file_path, os.path.join(dest_folder, filename))
        print(f"Routed to: {os.path.basename(dest_folder)}")

    # --- PROCESS VIDEOS ---
    elif file_ext in video_exts:
        unique_ids = set()
        has_group = False
        
        results_generator = model.track(
            source=file_path, 
            conf=CONF_THRESHOLD, 
            persist=True, 
            stream=True, 
            verbose=False
        )
        
        for result in results_generator:
            if result.boxes is None:
                continue
                
            classes = result.boxes.cls.int().tolist()
            
            # Optimization: If we spot a group, stop watching the video. We know where it belongs.
            if GROUP_CLASS_ID in classes:
                has_group = True
                break 
                
            # Log unique individual bats
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().tolist()
                for cls, track_id in zip(classes, track_ids):
                    if cls == BAT_CLASS_ID:
                        unique_ids.add(track_id)
                        
            # Secondary Optimization: If we hit 4 distinct bats, stop watching.
            if len(unique_ids) >= 4:
                break
                
        total_bats = len(unique_ids)
        dest_folder = determine_bucket(total_bats, has_group)
        
        # Move the entire original video file to the bucket
        shutil.copy(file_path, os.path.join(dest_folder, filename))
        print(f"Routed to: {os.path.basename(dest_folder)}")

    # --- MEMORY FLUSH ---
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# 5. CLEANUP
# ==========================================
if temp_workspace:
    print(f"\nCleaning up '{temp_workspace}'...")
    shutil.rmtree(temp_workspace)

print("\nTask complete! Your raw files are cleanly sorted in the 'sorted_media_files' directory.")
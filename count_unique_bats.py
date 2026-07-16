import os
import zipfile
import glob
import shutil
import gc
import torch
import cv2
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_DIR = "raw_videos_folder"      
OUTPUT_DIR = "sorted_media_files"
MODEL_PATH = "weights/best.pt"
CONF_THRESHOLD = 0.38
TEMP_WORKSPACE = "TEMPworkspace"

# Your YOLO Classes from the combined_data.yaml
BAT_CLASS_ID = 0
BAT_PART_CLASS_ID = 1
GROUP_CLASS_ID = 2

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

# Dynamically extract exact class IDs from the trained model's memory
name_to_id = {v: k for k, v in model.names.items()}
BAT_CLASS_ID = name_to_id.get('bat', 0)
BAT_PART_CLASS_ID = name_to_id.get('bat_part', 1)
GROUP_CLASS_ID = name_to_id.get('group_of_bat', 2)
print(f"Internal Model Class Map: {model.names}")

# ==========================================
# 3. SWEEP AND UNPACK INPUT DIRECTORY
# ==========================================
image_exts = ('.jpg', '.jpeg', '.png', '.JPG')
video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.MP4')
media_files = []

if not os.path.exists(INPUT_DIR):
    print(f"CRITICAL ERROR: The input folder '{INPUT_DIR}' does not exist.")
    exit(1)

os.makedirs(TEMP_WORKSPACE, exist_ok=True)
print(f"Scanning '{INPUT_DIR}' for media and zip files...")

# Walk through the raw_videos folder
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        file_ext = os.path.splitext(file)[1].lower()
        
        # If it's already a media file, queue it up
        if file_ext in image_exts or file_ext in video_exts:
            media_files.append(file_path)
            
        # If it's a zip file, crack it open
        elif file_ext == '.zip':
            print(f"  -> Extracting '{file}'...")
            zip_name = os.path.splitext(file)[0]
            # Create a unique subfolder so identically named files in different zips don't overwrite
            extract_path = os.path.join(TEMP_WORKSPACE, zip_name)
            os.makedirs(extract_path, exist_ok=True)
            
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                # Sweep the newly extracted folder for media
                for ext_root, _, ext_files in os.walk(extract_path):
                    for ext_file in ext_files:
                        e_path = os.path.join(ext_root, ext_file)
                        e_ext = os.path.splitext(ext_file)[1].lower()
                        if e_ext in image_exts or e_ext in video_exts:
                            media_files.append(e_path)
            except Exception as e:
                print(f"  -> Warning: Failed to extract {file} - {e}")

print(f"\nScan complete! Found {len(media_files)} total media files to sort.\n")

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
        
        if len(results[0].boxes) > 0:
            classes = results[0].boxes.cls.int().tolist()
            has_group = GROUP_CLASS_ID in classes
            bat_count = classes.count(BAT_CLASS_ID) + classes.count(BAT_PART_CLASS_ID)
            print(f"  -> Detected classes: {classes} | Calculated Bat Count: {bat_count}")
        else:
            has_group = False
            bat_count = 0
            print("  -> No detections above confidence threshold.")
            
        dest_folder = determine_bucket(bat_count, has_group)
        
        annotated_img = results[0].plot() 
        cv2.imwrite(os.path.join(dest_folder, filename), annotated_img)
        print(f"Routed to: {os.path.basename(dest_folder)}")

    # --- PROCESS VIDEOS ---
    elif file_ext in video_exts:
        unique_ids = set()
        has_group = False
        max_frame_bats = 0  # Fallback counter for untracked boxes
        
        # 1. Setup temporary holding path for the video
        temp_vid_path = os.path.join(TEMP_WORKSPACE, f"temp_{filename}")
        
        # 2. THIS WAS THE MISSING LINE: Start the YOLO tracker
        results_generator = model.track(source=file_path, conf=CONF_THRESHOLD, persist=False, stream=True, verbose=False, tracker="bytetrack.yaml")
        
        video_writer = None

        # 3. Read the video frame by frame
        for result in results_generator:
            if result.boxes is None:
                continue
            
            classes = result.boxes.cls.int().tolist()
            
            # Check for groups
            if GROUP_CLASS_ID in classes:
                has_group = True
                
            # Track unique IDs if the tracker successfully locked onto them
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().tolist()
                for cls, track_id in zip(classes, track_ids):
                    if cls == BAT_CLASS_ID or cls == BAT_PART_CLASS_ID:
                        unique_ids.add(track_id)
            
            # Fallback: Count total boxes in just this specific frame
            current_frame_bats = classes.count(BAT_CLASS_ID) + classes.count(BAT_PART_CLASS_ID)
            if current_frame_bats > max_frame_bats:
                max_frame_bats = current_frame_bats

            # 4. Initialize VideoWriter dynamically on the first frame
            if video_writer is None:
                h, w = result.orig_img.shape[:2]
                video_writer = cv2.VideoWriter(
                    temp_vid_path,
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h)
                )

            # 5. Draw boxes and write the annotated frame to the temp file
            annotated_frame = result.plot()
            video_writer.write(annotated_frame)
            
            # NOTE: We removed the 'break' optimization here. 
            # If we break early, your annotated video will be cut short!

        # 6. Close the video file
        if video_writer:
            video_writer.release()
            
        # 7. Determine final destination using whichever count is highest
        final_tally = max(len(unique_ids), max_frame_bats)
        print(f"  -> Video Tally: {len(unique_ids)} tracked | {max_frame_bats} max in one frame")
        
        dest_folder = determine_bucket(final_tally, has_group)
        
        # 8. Move the completely annotated video to the correct bucket
        final_output_path = os.path.join(dest_folder, filename)
        shutil.move(temp_vid_path, final_output_path)
        print(f"Routed to: {os.path.basename(dest_folder)}")

    # --- MEMORY FLUSH ---
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==========================================
# 5. CLEANUP
# ==========================================
print(f"\nCleaning up temporary zip extractions in '{TEMP_WORKSPACE}'...")
shutil.rmtree(TEMP_WORKSPACE)

print("\nTask complete! Your annotated files are cleanly sorted in the 'sorted_media_files' directory.")
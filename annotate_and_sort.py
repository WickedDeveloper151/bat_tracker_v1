import cv2
import os
import glob
import shutil
import zipfile
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "weights/best.pt"
INPUT_DIR = "raw_videos_folder"      
HAS_BATS_DIR = "sorted_has_bats"     
NO_BATS_DIR = "sorted_no_bats"       
TEMP_UNZIP_DIR = "temp_unpacked_data" # Temporary holding zone for zip contents

# Hardened Detection Guardrails
CONF_THRESHOLD = 0.55
MIN_BOX_SIZE = 10

# Create all necessary directories
for d in [HAS_BATS_DIR, NO_BATS_DIR, TEMP_UNZIP_DIR]:
    os.makedirs(d, exist_ok=True)

model = YOLO(MODEL_PATH)

# --- PHASE 1: ARCHIVE EXTRACTION ---
zip_files = glob.glob(os.path.join(INPUT_DIR, "*.zip"))
if zip_files:
    print(f"Detected {len(zip_files)} ZIP archives. Unpacking to temporary storage...")
    for zip_path in zip_files:
        zip_name = os.path.basename(zip_path).replace('.zip', '')
        extract_path = os.path.join(TEMP_UNZIP_DIR, zip_name)
        os.makedirs(extract_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"  -> Unpacked {os.path.basename(zip_path)}")

# --- PHASE 2: FILE GATHERING ---
# Look through both the original raw folder AND the temporary unpacked folder
search_dirs = [INPUT_DIR, TEMP_UNZIP_DIR]
all_files = []

for d in search_dirs:
    # Use recursive search in case the ZIP had folders inside it
    all_files.extend(glob.glob(os.path.join(d, "**", "*.*"), recursive=True))

# Filter for media we actually care about
video_exts = {".mp4", ".avi", ".mkv", ".mov", ".MP4"}
image_exts = {".jpg", ".jpeg", ".png" , ".JPG"}

media_files = [f for f in all_files if os.path.splitext(f)[1].lower() in (video_exts | image_exts)]
print(f"\nFound {len(media_files)} total media files to process...\n")

# --- PHASE 3: PROCESSING ENGINE ---
for file_path in media_files:
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    
    # Setup temporary output path for the annotated version
    temp_output_path = os.path.join(NO_BATS_DIR, filename)
    print(f"Processing {filename}...")

    # ==========================================
    # BRANCH A: IMAGE PROCESSING
    # ==========================================
    if ext in image_exts:
        img = cv2.imread(file_path)
        if img is None:
            continue
            
        results = model.predict(img, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
        bat_found = False
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE:
                bat_found = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_score = float(box.conf[0])
                cv2.putText(img, f"Bat {conf_score:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the annotated image
        cv2.imwrite(temp_output_path, img)
        
        if bat_found:
            shutil.move(temp_output_path, os.path.join(HAS_BATS_DIR, filename))
            print(f"  -> IMAGE MATCH: Saved to {HAS_BATS_DIR}/")
        else:
            print(f"  -> NO MATCH: Saved to {NO_BATS_DIR}/")

    # ==========================================
    # BRANCH B: VIDEO PROCESSING
    # ==========================================
    elif ext in video_exts:
        cap = cv2.VideoCapture(file_path)
        width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
        
        out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frames_with_bats = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
                
            results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
            frame_had_detection = False
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE:
                    frame_had_detection = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Bat {float(box.conf[0]):.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if frame_had_detection: frames_with_bats += 1
            out.write(frame)
            
        cap.release()
        out.release()
        
        # Temporal Filter: Needs >1 frame to prove it isn't a glitch
        if frames_with_bats > 1:
            shutil.move(temp_output_path, os.path.join(HAS_BATS_DIR, filename))
            print(f"  -> VIDEO MATCH ({frames_with_bats} frames): Saved to {HAS_BATS_DIR}/")
        else:
            print(f"  -> NO MATCH: Saved to {NO_BATS_DIR}/")

# --- PHASE 4: CLEANUP ---
# Wipe the temporary unzipped data to save disk space
if os.path.exists(TEMP_UNZIP_DIR):
    shutil.rmtree(TEMP_UNZIP_DIR)
    print("\nTemporary extraction folders cleaned up.")

print("All files processed, annotated, and sorted successfully!")
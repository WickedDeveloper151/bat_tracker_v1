import os
import tarfile
from ultralytics import YOLO
import shutil

# ==========================================
# 0. DATA PREPARATION (Extracting the Archive)
# ==========================================
# The tar file we generated from the prep script
TAR_FILE_PATH = "ready_for_training.tar" 
EXTRACT_DESTINATION = ""

# The prep script packs everything into a folder named "yolo_dataset"
EXPECTED_DATASET_FOLDER = "yolo_dataset"

print("--- Checking Dataset Status ---")
if os.path.exists(EXPECTED_DATASET_FOLDER):
    print("Old dataset folder found. Deleting it to ensure we extract the newest data...")
    shutil.rmtree(EXPECTED_DATASET_FOLDER)

print(f"Extracting '{TAR_FILE_PATH}'...")
try:
    with tarfile.open(TAR_FILE_PATH, "r:*") as tar:
        tar.extractall(path=EXTRACT_DESTINATION)
    print("Extraction complete!")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to extract tar file. {e}")
    exit(1)

# ==========================================
# 1. LOAD THE MODEL
# ==========================================
# Load your existing hardened brain
model = YOLO("weights/best.pt") 

print("\nSuccessfully loaded existing weights. Beginning fine-tuning on OSC...")

# ==========================================
# 2. TRAIN THE MODEL
# ==========================================
# Train on the merged and balanced dataset
results = model.train(
    # CRITICAL FIX: Point directly to the yaml inside the yolo_dataset folder
    data="yolo_dataset/data.yaml",
    epochs=3,           
    imgsz=640,
    patience=50,
    
    # --- KNOWLEDGE PRESERVATION ---
    lr0=0.001,            # Low learning rate to protect old single-bat memory
    freeze=10, # Lock the foundational vision layers
    
    # --- CAVE-SPECIFIC AUGMENTATION ---
    degrees=15.0,         
    translate=0.1,        
    scale=0.5,            
    fliplr=0.5,           
    flipud=0.5,           # Crucial for bats hanging on ceilings
    hsv_h=0.015,            
    hsv_s=0.2,            
    hsv_v=0.4,            # Simulates thermal washout
    mosaic=1.0,           
    erasing=0.2,          # Forces AI to learn partially hidden bats
    
    project="bat_tracking_project",
    name="hybrid_model")

print("\nFine-tuning complete! Your upgraded model is in 'bat_tracking_project/hybrid_model_v1/weights/'.")
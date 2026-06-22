# Automated Bat Tracking & Sorting Pipeline

This tool uses an artificial intelligence (AI) model to automatically watch hours of infrared cave footage or scan hundreds of trail-camera photos. 
It finds the bats, draws tracking boxes around them, and sorts your files into clean, organized folders so you don't have to watch empty footage.
This guide will walk you through exactly how to use it.

What is actually inside this folder?
- raw_videos_folder/: This is your drop-zone. You will put all your un-watched videos, photos, or .zip files in here.
- weights/: This contains a file called best.pt. Think of this as the "AI's Brain." It is the file that actually knows what a bat looks like.
- annotate_and_sort.py: This is the actual engine. It's the script that wakes the AI up and tells it to look at your videos.
- requirements.txt: A simple text list of the background software the AI needs to run (like the code that understands video files).

# Step 1: First-Time Setup (You only do this once!)
To run this tool, your computer needs to have Python installed. Python is just the language the tool is written in.

A. Check for Python
1. On Windows: Click your Start button, type cmd, and hit Enter to open the Command Prompt (a black text window).

On Mac: Press Command + Space, type Terminal, and hit Enter.

2. Type this exact command and press Enter:
python --version
(If it spits out a number like "Python 3.10", you are good to go! If it says "command not found," download and install Python from Python.org).


B. Create the Virtual Environment and Install the AI Background Software
Your computer needs to download a few free background tools to read the videos.

1. Open your Terminal or Command Prompt.

2. Navigate your terminal to the folder that this Readme file is in

3. Create the virtual environment by typing this and hitting Enter:
python -m venv bat_env

4. Activate the environment:

 - On Windows: Type bat_env\Scripts\activate and hit Enter.

 - On Mac: Type source bat_env/bin/activate and hit Enter.
(You will know it worked if you see (bat_env) pop up at the beginning of your text line!)


5. Type this exact command and hit Enter:
pip install -r requirements.txt

6. Let it download. It might take a few minutes. Once it finishes, you never have to do this step again!

Important Note for the Future: Every time you close the command prompt and come back the next day to process more videos, you must repeat Step 3 to reactivate the environment before you run the tool!




# Step 2: Running the Tool
Whenever you have new data from the field, follow these three simple steps:

1. Load Your Data
Drag and drop your raw videos (.mp4, .avi), your photos (.jpg, .png), or even your compressed .zip files directly into the raw_videos_folder.

2. Start the Engine
Open your Terminal or Command Prompt, make sure your (bat_env) environment is active, and type:
python 	annotate_and_sort.py

Press Enter.


3. Let it Work
The screen will start printing out text, letting you know exactly what it is doing.

- If it finds .zip files, it will automatically unpack them for you.

- It will process every file, and it will tell you if it found a bat or not.

- Note: Depending on how many videos you added, this could take anywhere from a few minutes to a few hours.



# Step 3: Checking Your Results
When the text on the screen says "All files processed, annotated, and sorted successfully!", the tool is done.

If you look inside your main folder, you will notice the tool automatically created two brand new folders for you:
- sorted_has_bats/: Every video or photo in this folder is guaranteed to have at least one bat in it. Furthermore, the AI has drawn a green box around the bat with a confidence score (e.g., 0.85 means it is 85% sure it is a bat).
-sorted_no_bats/: Everything in here is empty footage of cave walls. You can safely ignore this folder.


Special Instructions for Supercomputer (HPC) Users
If you are running this on a research cluster rather than a personal laptop, skip parts  1 and 2  of Step 2. Instead:

1. Upload your data into the raw_videos_folder/.

2. Edit run_master_pipeline.sh by channging this section 
# --- UC ARC SPECIFIC CHANGES ---
#SBATCH --account=YOUR_UC_PROJECT_ID   # Replace with UC billing account
#SBATCH --partition=gpu                # Replace with the exact UC GPU partition name

#Load the exact CUDA module name UC uses (Run 'module avail cuda' to find it)
module load cuda/xx.x.x  

3. Open your cluster terminal and type sbatch run_master_pipeline.sh.

4. The cluster will handle the rest! You can check the .log files that appear to see its progress.







import cv2
import os
import sys
#usage: python FrameExtractor.py video_path -1 -1
def parse_time(time_str):
    """
    Converts MM:SS format or raw seconds string to total seconds (integer).
    Returns None if the format is invalid or '-1'.
    """
    if time_str == "-1":
        return None
    try:
        if ':' in time_str:
            # Split minutes and seconds and calculate total
            mm, ss = map(int, time_str.split(':'))
            return mm * 60 + ss
        # If no colon, assume the input is already in seconds
        return int(time_str)
    except ValueError:
        return None

def main():
    # Ensure a video path is provided via command line
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <videoPath> [start MM:SS] [end MM:SS]")
        return

    video_file = sys.argv[1]

    # Extract the base filename (without extension) to name the output directories
    base_name = os.path.splitext(os.path.basename(video_file))[0]

    # Define output paths for Left (L) and Right (R) eye crops
    
    dir_r = f"./data/groundtruth/{base_name}/frames/output_{base_name}_R"
    dir_l = f"./data/groundtruth/{base_name}/frames/output_{base_name}_L"
    dir_full = f"./data/groundtruth/{base_name}/frames/output_{base_name}_FULL"

    # Create directories if they don't already exist
    os.makedirs(dir_r, exist_ok=True)
    os.makedirs(dir_l, exist_ok=True)
    os.makedirs(dir_full, exist_ok=True)

    # Default processing range (from start to the end of the video)
    start_time_sec = 0
    end_time_sec = float('inf')

    # Parse optional start and end times from command line arguments
    if len(sys.argv) > 2:
        t = parse_time(sys.argv[2])
        if t is not None: start_time_sec = t

    if len(sys.argv) > 3:
        t = parse_time(sys.argv[3])
        if t is not None: end_time_sec = t

    # Initialize video capture
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video metadata: Frames Per Second and Total Frame Count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate starting frame and jump to it
    start_frame = int(start_time_sec * fps)
    if start_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    # Downsampling logic (Skip frames if needed)
    # samples_per_second = -1 means process every single frame
    samples_per_second = -1
    sample_step = 1 if samples_per_second == -1 else max(1, int(fps / samples_per_second))

    print(f"Processing frames from {start_time_sec}s to {end_time_sec}s...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video or read error

        # Check if the current time exceeds the specified end time
        current_second = current_frame / fps
        if current_second > end_time_sec:
            break

        # Process the frame based on the sample step
        if current_frame % sample_step == 0:
            height, width = frame.shape[:2]
            middle = width // 2

            # Split the frame vertically into two halves (Stereoscopic/Eye split)
            # Numpy slicing: [y_start:y_end, x_start:x_end]
            left_eye = frame[:, 0:middle]
            right_eye = frame[:, middle:width]

            # Generate filenames with 7-digit zero padding for easy sorting
            name_l = os.path.join(dir_l, f"{base_name}_frame_{current_frame:07d}_L.png")
            name_r = os.path.join(dir_r, f"{base_name}_frame_{current_frame:07d}_R.png")

            # Save the cropped images
            cv2.imwrite(name_l, left_eye)
            cv2.imwrite(name_r, right_eye)

            # Also save the full frame
            name_full = os.path.join(dir_full, f"{base_name}_frame_{current_frame:07d}_FULL.png")
            cv2.imwrite(name_full, frame)

            # Print progress every 500 frames
            if current_frame % 500 == 0:
                print(f"Processed frame: {current_frame}")

        current_frame += 1

    # Cleanup resources
    cap.release()
    print("Process completed successfully.")

if __name__ == "__main__":
    main()

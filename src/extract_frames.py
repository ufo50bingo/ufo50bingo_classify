"""
ufo50_frame_extractor.py

- Downloads YouTube videos
- Extracts frames at a specified FPS
- Saves frames in structured folders
- Resizes for model input if needed
"""

import os
import cv2
from tqdm import tqdm
import yt_dlp
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEOS_DIR = "videos"  # downloaded videos
FRAMES_DIR = "frames"  # where frames are saved
TARGET_SIZE = (320, 180)  # model input size (square)
FPS_EXTRACT = 2  # frames per second to extract
VIDEO_LIST = [
    # Example: ("URL", "game_name")
    ("https://www.youtube.com/watch?v=9FWVqxpuh5I", "barbuta"),
    ("https://www.youtube.com/watch?v=vKfa9Ni9P3k", "bughunter"),
]


# -----------------------------
# FUNCTIONS
# -----------------------------
def download_youtube_video(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ydl_opts = {
        "format": "mp4",  # download mp4
        "outtmpl": save_path,  # save path
        "noplaylist": True,  # just the video, not playlist
        "quiet": False,  # show progress
        "merge_output_format": "mp4",  # ensures final file is mp4
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def extract_frames(video_path, output_dir, fps_extract=2, target_size=None):
    """
    Extract frames from a video at fps_extract frames per second.
    Optionally resize to target_size (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # fallback if unknown

    frame_interval = max(int(video_fps / fps_extract), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved_count = 0
    frame_count = 0

    pbar = tqdm(
        total=total_frames,
        desc=f"Extracting frames from {os.path.basename(video_path)}",
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count % frame_interval == 0:
            if target_size:
                # Resize using nearest neighbor to preserve pixel art
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)

            # Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")


# -----------------------------
# MAIN PROCESS
# -----------------------------
def main():
    for url, game_name in VIDEO_LIST:
        video_save_path = os.path.join(VIDEOS_DIR, f"{game_name}.mp4")
        download_youtube_video(url, video_save_path)

        output_frames_dir = os.path.join(FRAMES_DIR, game_name)
        extract_frames(
            video_save_path,
            output_frames_dir,
            fps_extract=FPS_EXTRACT,
            target_size=TARGET_SIZE,
        )


if __name__ == "__main__":
    main()

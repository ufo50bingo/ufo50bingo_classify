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
    ("https://www.youtube.com/watch?v=LnnKUHltiGQ", "ninpek"),
    ("https://www.youtube.com/watch?v=Hj6rFJR64hw", "paintchase"),
    ("https://www.youtube.com/watch?v=gMNsmi0YKGk", "magicgarden"),
    ("https://www.youtube.com/watch?v=uy7grjuo7hY", "mortol"),
    # remove first bit
    ("https://www.youtube.com/watch?v=ffRdK5C8sEs", "velgress"),
    # remove first bit
    ("https://www.youtube.com/watch?v=JJym0J9PSO4", "planetzoldath"),
    ("https://www.youtube.com/watch?v=TwK7fjktpec", "attactics"),
    # remove first bit
    ("https://www.youtube.com/watch?v=Yw6-lA9ZPk8", "devilition"),
    # trim
    ("https://www.youtube.com/watch?v=3MHi9SnkT6g", "kickclubboy"),
    ("https://www.youtube.com/watch?v=fcZ-8tz5aM0", "kickclubgirl"),
    # trim plus maybe add custom screen?
    ("https://www.youtube.com/watch?v=wMoRLx45q2w", "avianos"),
    # trim
    ("https://www.youtube.com/watch?v=m8i_iXnupZo", "mooncat"),
    # trim
    ("https://www.youtube.com/watch?v=ZdNQVZri38Av", "bushidoball"),
    # trim
    # need custom levels for BK
    ("https://www.youtube.com/watch?v=dDxKfOatrTk", "blockkoala"),
    ("https://www.youtube.com/watch?v=0SGgCFcMEF0", "camouflage"),
    ("https://www.youtube.com/watch?v=TpyjsQfZi7Q", "campanella"),
    ("https://www.youtube.com/watch?v=mT9Ycv3yBtk", "golfaria"),
    ("https://www.youtube.com/watch?v=YH1oNyg5R2s", "thebigbellrace"),
    ("https://www.youtube.com/watch?v=wWEVRwu0IWc", "warptank"),
    ("https://www.youtube.com/watch?v=V2DhzYefBnQ", "waldorfsjourney"),
    ("https://www.youtube.com/watch?v=RcX9fsIc6bI", "porgy"),
    ("https://www.youtube.com/watch?v=3ICUvWcAvUo", "oniondelivery"),
    ("https://www.youtube.com/watch?v=eyxzLIMYA3M", "caramelcaramel"),
    ("https://www.youtube.com/watch?v=BsyoLOTWu8E", "partyhouse"),
    ("https://www.youtube.com/watch?v=24IGMFHbDfI", "hotfoot"),
    ("https://www.youtube.com/watch?v=lgbS9q6G1ew", "divers"),
    ("https://www.youtube.com/watch?v=s9YshoDce_4", "railheist"),
    # remember to remove pause screen
    ("https://www.youtube.com/watch?v=R9YvcbyRNT0", "vainger"),
    # lots to trim
    ("https://www.youtube.com/watch?v=SdtvA9MKCV0", "rockonisland"),
    ("https://www.youtube.com/watch?v=Pf_LQc92axo", "pingolf"),
    ("https://www.youtube.com/watch?v=3HtV6ORsf6s", "mortolii"),
    ("https://www.youtube.com/watch?v=Qlux9aPCwBM", "fisthelljay"),
    ("https://www.youtube.com/watch?v=qh1EYLUXbWM", "fisthellcat"),
    ("https://www.youtube.com/watch?v=XJ2YjozthTc", "fisthellvictor"),
    ("https://www.youtube.com/watch?v=HEADmWR9c_o", "fisthellamy"),
    ("https://www.youtube.com/watch?v=DiypNQGxEUs", "fisthellgym"),
    ("https://www.youtube.com/watch?v=jsn-SKbwRZ4", "overbold"),
    ("https://www.youtube.com/watch?v=QTiqmvUfpu0", "campanella2-a"),
    ("https://www.youtube.com/watch?v=XB32JXs6OAg", "campanella2-b"),
    ("https://www.youtube.com/watch?v=sbl6-uvUm_g", "hypercontender"),
    ("https://www.youtube.com/watch?v=dvR6lxFRRSE", "valbrace"),
    ("https://www.youtube.com/watch?v=VPdLhcaxGTU", "rakshasa"),
    ("https://www.youtube.com/watch?v=w-4KQhFaKNc", "starwaspir"),
    ("https://www.youtube.com/watch?v=y8v1suMNuys", "grimstone"),
    ("https://www.youtube.com/watch?v=CLqGuMa_C1Q", "lordsofdiskonia"),
    ("https://www.youtube.com/watch?v=eTDku8OjnXY", "nightmanor"),
    ("https://www.youtube.com/watch?v=6C9yVXxQ8MM", "elfazarshat"),
    ("https://www.youtube.com/watch?v=uBsweRtrDko", "pilotquest"),
    ("https://www.youtube.com/watch?v=nYO8YfOacL8", "miniandmax"),
    ("https://www.youtube.com/watch?v=7h4Vz4QdYxc", "combatants"),
    ("https://www.youtube.com/watch?v=LyDdtBbwQA4", "quibblerace"),
    ("https://www.youtube.com/watch?v=P7gxwzankCw", "seasidedrive"),
    ("https://www.youtube.com/watch?v=z8IF2QeCzyo", "campanella3"),
    ("https://www.youtube.com/watch?v=HJLT3tgJJ8I", "cyberowls"),
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

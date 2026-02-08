import cv2
import numpy as np

def create_video_from_image(image_path, video_path, duration=5, fps=24):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return

    height, width, layers = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for _ in range(duration * fps):
        video.write(img)

    video.release()
    print(f"Video created: {video_path}")

create_video_from_image('1f364239b96c02334cdcafae7b3e2dcb.jpg', 'enhanced_realism.mp4', duration=2)
create_video_from_image('ea70c55e8b3e30131db34a7c493883ac.jpg', 'enhanced_segmented_5s.mp4', duration=5)

import cv2
import os

def check_video(filepath, expected_min_frames=1):
    if not os.path.exists(filepath):
        print(f"FAILED: {filepath} does not exist")
        return False

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"FAILED: {filepath} could not be opened")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count < expected_min_frames:
        print(f"FAILED: {filepath} has only {frame_count} frames, expected at least {expected_min_frames}")
        return False

    print(f"PASSED: {filepath} ({width}x{height}, {frame_count} frames)")
    return True

def check_image(filepath):
    if not os.path.exists(filepath):
        print(f"FAILED: {filepath} does not exist")
        return False

    img = cv2.imread(filepath)
    if img is None:
        print(f"FAILED: {filepath} could not be read")
        return False

    h, w, c = img.shape
    print(f"PASSED: {filepath} ({w}x{h}, {c} channels)")
    return True

print("Validating outputs...")
v1 = check_video("enhanced_realism_AI_COLORIZED.mp4", 10)
v2 = check_video("enhanced_segmented_5s_AI_COLORIZED.mp4", 100) # 5s at 24fps = 120 frames
i1 = check_image("enhanced_realism_comparison.jpg")
i2 = check_image("enhanced_segmented_5s_comparison.jpg")

if all([v1, v2, i1, i2]):
    print("\nALL VALIDATIONS PASSED")
else:
    print("\nSOME VALIDATIONS FAILED")
    exit(1)

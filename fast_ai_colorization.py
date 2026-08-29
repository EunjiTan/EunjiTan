import cv2
import numpy as np
import argparse
import os
import sys

class AIColorizer:
    def __init__(self, palette='natural'):
        self.palette = palette
        # LAB values (OpenCV relative to 128 for A and B)
        # L is not used as we preserve original luminance
        self.palettes = {
            'natural': {
                'sky': [0, -5, -25],
                'vegetation': [0, -25, 20],
                'ground': [0, 5, 15],
                'skin': [0, 15, 10],
                'buildings': [0, 0, 0]
            },
            'warm': {
                'sky': [0, 5, 40],
                'vegetation': [0, -10, 50],
                'ground': [0, 15, 30],
                'skin': [0, 20, 20],
                'buildings': [0, 5, 10]
            },
            'cinematic': {
                'sky': [0, -10, -30],
                'vegetation': [0, -15, 5],
                'ground': [0, 0, 0],
                'skin': [0, 5, 5],
                'buildings': [0, -5, -10]
            }
        }
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def colorize_frame(self, frame):
        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast enhancement
        gray = self.clahe.apply(gray)

        # Bilateral filter for denoising while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # LAB approach
        h, w = gray.shape
        l_channel = gray.astype(np.float32)
        a_channel = np.zeros((h, w), dtype=np.float32)
        b_channel = np.zeros((h, w), dtype=np.float32)

        # Semantic masks based on vertical position
        sky_mask = np.zeros((h, w), dtype=np.float32)
        sky_mask[:int(h*0.4), :] = 1.0

        ground_mask = np.zeros((h, w), dtype=np.float32)
        ground_mask[int(h*0.6):, :] = 1.0

        middle_mask = np.zeros((h, w), dtype=np.float32)
        middle_mask[int(h*0.4):int(h*0.6), :] = 1.0

        # Semantic masks based on brightness
        bright_mask = (l_channel > 180).astype(np.float32)
        dark_mask = (l_channel < 80).astype(np.float32)

        p = self.palettes[self.palette]

        # Apply colors to LAB channels
        a_channel += sky_mask * p['sky'][1]
        b_channel += sky_mask * p['sky'][2]

        veg_mask = ground_mask * (1 - dark_mask)
        a_channel += veg_mask * p['vegetation'][1]
        b_channel += veg_mask * p['vegetation'][2]

        g_mask = ground_mask * dark_mask
        a_channel += g_mask * p['ground'][1]
        b_channel += g_mask * p['ground'][2]

        skin_mask = middle_mask * bright_mask
        a_channel += skin_mask * p['skin'][1]
        b_channel += skin_mask * p['skin'][2]

        neutral_mask = middle_mask * (1 - bright_mask)
        a_channel += neutral_mask * p['buildings'][1]
        b_channel += neutral_mask * p['buildings'][2]

        # Blur the A and B channels for smooth transitions between regions
        a_channel = cv2.GaussianBlur(a_channel, (101, 101), 0)
        b_channel = cv2.GaussianBlur(b_channel, (101, 101), 0)

        # Reconstruct LAB (OpenCV LAB scale is L: 0-255, A: 128 neutral, B: 128 neutral)
        lab = cv2.merge([l_channel, a_channel + 128, b_channel + 128])
        lab = np.clip(lab, 0, 255).astype(np.uint8)

        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Saturation Boost in HSV space
        hsv = cv2.cvtColor(colorized, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.4 # Slightly increased boost
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        colorized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Final Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        colorized = cv2.filter2D(colorized, -1, kernel)

        return colorized

def process_video(input_path, output_path, colorizer):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 24

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        colorized = colorizer.colorize_frame(frame)
        out.write(colorized)

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Fast AI Colorization (Heuristic-based Semantic Pipeline)")
    parser.add_argument("input", help="Input image or video file")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--palette", default="natural", choices=["natural", "warm", "cinematic"])
    parser.add_argument("--comparison", action="store_true", help="Generate comparison image")
    args = parser.parse_args()

    colorizer = AIColorizer(palette=args.palette)

    if args.input.lower().endswith(('.mp4', '.avi', '.mov')):
        if not args.output:
            args.output = args.input.rsplit('.', 1)[0] + "_AI_COLORIZED.mp4"
        process_video(args.input, args.output, colorizer)
        print(f"Processed video saved to {args.output}")
    else:
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"Error: Could not read image {args.input}")
            return

        colorized = colorizer.colorize_frame(frame)

        if args.comparison:
            h, w, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            comparison = np.hstack((gray_bgr, colorized))
            if not args.output:
                args.output = args.input.rsplit('.', 1)[0] + "_comparison.jpg"
            cv2.imwrite(args.output, comparison)
            print(f"Comparison image saved to {args.output}")
        else:
            if not args.output:
                args.output = args.input.rsplit('.', 1)[0] + "_AI_COLORIZED.jpg"
            cv2.imwrite(args.output, colorized)
            print(f"Colorized image saved to {args.output}")

if __name__ == "__main__":
    main()

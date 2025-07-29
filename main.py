import copy
import time
import argparse
import cv2
import math
from pathlib import Path
import os
import glob
import onnxruntime
import numpy as np

# --- Configuration for Your Custom Model ---
CLASS_NAMES = {0: 'unripe', 1: 'ripe', 2: 'semi-ripe'}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 255)}

# --- StrawberryDetector Class (Unchanged) ---
class StrawberryDetector:
    def __init__(self, model_path, use_gpu=False):
        self.model_path = model_path
        providers = ['CUDAExecutionProvider'] if use_gpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(f"🍓 Strawberry Detector Initialized with model: {model_path} on {self.session.get_providers()[0]}")
    def __call__(self, frame):
        img_height, img_width, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_rgb, (self.input_width, self.input_height))
        img_tensor = resized_img.transpose(2, 0, 1)
        img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
        img_tensor = img_tensor.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: img_tensor})
        predictions = np.squeeze(outputs[0]).T
        conf_threshold = 0.4
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold]
        scores = scores[scores > conf_threshold]
        if len(predictions) == 0: return [], [], []
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        x, y, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
        x1 = (x - w / 2) / self.input_width * img_width; y1 = (y - h / 2) / self.input_height * img_height
        x2 = (x + w / 2) / self.input_width * img_width; y2 = (y + h / 2) / self.input_height * img_height
        bboxes = np.column_stack((x1, y1, x2, y2))
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), conf_threshold, 0.5)
        if len(indices) == 0: return [], [], []
        return bboxes[indices], scores[indices], class_ids[indices]

from Tracker.tracker import MultiObjectTracker

# --- get_args Function (Unchanged) ---
def get_args():
    parser = argparse.ArgumentParser(description="Strawberry Ripeness Tracking with ByteTrack")
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the input video file OR a directory containing video files.')
    parser.add_argument('--output_dir', type=str, default='output_videos',
                        help='Directory where the processed videos will be saved.')
    parser.add_argument('--tracker', choices=['motpy', 'mc_bytetrack', 'mc_norfair', 'mc_sort', 'mc_deepsort'], default='mc_bytetrack')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    return args

# --- main Function for Batch Processing ---
def main():
    args = get_args()
    input_path = args.input
    output_dir = args.output_dir
    tracker_name = args.tracker
    use_gpu = args.use_gpu

    os.makedirs(output_dir, exist_ok=True)

    video_files = []
    if os.path.isfile(input_path):
        video_files.append(input_path)
    elif os.path.isdir(input_path):
        print(f"Scanning directory for videos: {input_path}")
        for ext in ('*.mp4', '*.mov', '*.avi', '*.mkv'):
            video_files.extend(glob.glob(os.path.join(input_path, ext)))
    
    if not video_files:
        print(f"Error: No valid video files found at path: {input_path}")
        return

    print(f"Found {len(video_files)} video(s) to process.")

    detector = StrawberryDetector(model_path='custom_models/best.onnx', use_gpu=use_gpu)

    # --- NEW: List to store the average FPS of each video ---
    all_videos_fps_list = []

    for video_path in video_files:
        print(f"\n--- Processing video: {os.path.basename(video_path)} ---")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}. Skipping.")
            continue
        
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        tracker = MultiObjectTracker(tracker_name, cap_fps, use_gpu=use_gpu)
        
        filtered_ids_dict = {0: set(), 1: set(), 2: set()}

        ret, img = cap.read()
        if not ret:
            print(f"Error: Could not read the first frame of {video_path}. Skipping.")
            continue
            
        h, w, _ = img.shape
        
        input_p = Path(video_path)
        output_filename = f"processed_{input_p.stem}.mp4"
        save_path = os.path.join(output_dir, output_filename)

        vid_writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), cap_fps, (w, h))

        # --- NEW: List to store time for each frame in THIS video ---
        frame_processing_times = []

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            debug_image = copy.deepcopy(frame)
            frame_h, frame_w, _ = debug_image.shape
            x_min = int(frame_w / 2 - 250)
            x_max = int(frame_w / 2 + 250)

            d_bboxes, d_scores, d_class_ids = detector(frame)
            track_ids, t_bboxes, t_scores, t_class_ids = tracker(frame, d_bboxes, d_scores, d_class_ids)
            filtered_ids_dict = total_count(track_ids, t_bboxes, t_class_ids, x_min, x_max, filtered_ids_dict)
            
            total_numbers = {
                CLASS_NAMES[0]: len(filtered_ids_dict[0]),
                CLASS_NAMES[1]: len(filtered_ids_dict[1]),
                CLASS_NAMES[2]: len(filtered_ids_dict[2]),
            }

            elapsed_time = time.time() - start_time
            # --- NEW: Store the time for this frame ---
            frame_processing_times.append(elapsed_time)

            debug_image = draw_debug_info(
                debug_image, elapsed_time, track_ids, t_bboxes, t_scores,
                t_class_ids, total_numbers, x_min, x_max, frame_h
            )

            key = cv2.waitKey(1)
            if key == 27:
                break
            cv2.imshow('Strawberry Ripeness Tracking', debug_image)
            vid_writer.write(debug_image)

        # --- AFTER a single video finishes ---
        vid_writer.release()
        cap.release()
        print(f"✅ Processing complete. Video saved to: {save_path}")
        
        # --- NEW: Calculate and print the statistics for the video ---
        if len(frame_processing_times) > 20: # Ensure there are enough frames
            stable_times = frame_processing_times[20:] # Exclude first 20 warm-up frames
            avg_time_per_frame = np.mean(stable_times)
            avg_fps = 1.0 / avg_time_per_frame
            print("-" * 40)
            print(f"Video Statistics for: {os.path.basename(video_path)}")
            print(f"  Average FPS (excluding warm-up): {avg_fps:.2f}")
            print("-" * 40)
            all_videos_fps_list.append(avg_fps) # Add this video's FPS to the master list
    
    # --- AFTER all videos have been processed ---
    cv2.destroyAllWindows()
    
    # --- NEW: Calculate and print the overall statistics ---
    if all_videos_fps_list:
        overall_avg_fps = np.mean(all_videos_fps_list)
        overall_std_dev_fps = np.std(all_videos_fps_list)
        print("\n" + "=" * 50)
        print("           OVERALL SYSTEM PERFORMANCE           ")
        print("=" * 50)
        print(f"  Tracker: {tracker_name}")
        print(f"  Average FPS across all {len(all_videos_fps_list)} videos: {overall_avg_fps:.2f}")
        print(f"  Standard Deviation of FPS: {overall_std_dev_fps:.2f}")
        print("=" * 50)
        print("\nUse the 'Average FPS across all videos' value for your paper.")


# --- total_count function (Unchanged) ---
def total_count(track_ids, bboxes, class_ids, x_min, x_max, filtered_ids_dict):
    for i in range(len(bboxes)):
        x1, _, x2, _ = bboxes[i]; track_id = track_ids[i]; class_id = int(class_ids[i])
        x_center = int(x1 + ((x2 - x1) * 0.5))
        if x_min < x_center < x_max:
            if class_id in filtered_ids_dict:
                filtered_ids_dict[class_id].add(track_id)
    return filtered_ids_dict

# --- draw_debug_info function (Unchanged) ---
def draw_debug_info(debug_image, elapsed_time, track_ids, bboxes, scores, class_ids, count_dict, x_min, x_max, frame_h):
    frame_w = debug_image.shape[1]
    overlay = debug_image.copy()
    panel_x, panel_y, panel_w, panel_h = 10, 10, 320, 140
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    alpha = 0.6
    debug_image = cv2.addWeighted(overlay, alpha, debug_image, 1 - alpha, 0)
    roi_overlay = debug_image.copy()
    ROI_COLOR = (200, 100, 0)
    cv2.rectangle(roi_overlay, (x_min, 0), (x_max, frame_h), ROI_COLOR, -1)
    roi_alpha = 0.2
    debug_image = cv2.addWeighted(roi_overlay, roi_alpha, debug_image, 1 - roi_alpha, 0)
    cv2.line(debug_image, (x_min, 0), (x_min, frame_h), ROI_COLOR, 2)
    cv2.line(debug_image, (x_max, 0), (x_max, frame_h), ROI_COLOR, 2)
    for track_id, bbox, score, class_id in zip(track_ids, bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        class_id = int(class_id); color = CLASS_COLORS.get(class_id, (255, 255, 255)); class_name = CLASS_NAMES.get(class_id, "Unknown")
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, thickness=2)
        text = f"{class_name} (ID:{track_id}) {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(debug_image, (x1, y1 - text_height - 8), (x1 + text_width, y1), color, -1)
        cv2.putText(debug_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    text_color = (255, 255, 255)
    cv2.putText(debug_image, "Statistics", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(debug_image, f"FPS: {fps:.1f} ({elapsed_time * 1000:.1f}ms)", (panel_x + 10, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
    y_offset = panel_y + 85
    for class_id, name in CLASS_NAMES.items():
        count = count_dict.get(name, 0); color = CLASS_COLORS[class_id]; count_text = f"Count ({name}): {count}"
        cv2.putText(debug_image, count_text, (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
    return debug_image

if __name__ == '__main__':
    main()
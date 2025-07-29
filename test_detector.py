import cv2
import os
import numpy as np
import argparse
from pathlib import Path

# --- Part 1: Configuration and Detector Class (No changes needed here) ---

# Define class names and corresponding BGR colors for drawing.
CLASS_NAMES = {0: 'unripe', 1: 'ripe', 2: 'semi-ripe'}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 255)}

# Custom StrawberryDetector class to handle ONNX model loading and inference.
import onnxruntime

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
        x1 = (x - w / 2) / self.input_width * img_width
        y1 = (y - h / 2) / self.input_height * img_height
        x2 = (x + w / 2) / self.input_width * img_width
        y2 = (y + h / 2) / self.input_height * img_height
        bboxes = np.column_stack((x1, y1, x2, y2))
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), conf_threshold, 0.5)
        if len(indices) == 0: return [], [], []
        return bboxes[indices], scores[indices], class_ids[indices]

# --- Part 2: Visualization Helper Function ---

def draw_strawberry_inferences(image, bboxes, scores, class_ids):
    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        class_id = int(class_id)
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        
        text = f"{class_name}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    return image

# --- Part 3: Main Execution Block ---

def main():
    # Set up argument parser for command-line use
    parser = argparse.ArgumentParser(description="Test a strawberry detector ONNX model on a single image.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model', type=str, default='custom_models/best.onnx', help='Path to the ONNX model file.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference if available.')
    args = parser.parse_args()

    # 1. Check if files exist
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found at {args.model}")
        return
    if not os.path.exists(args.image):
        print(f"ERROR: Image file not found at {args.image}")
        return

    # 2. Load Detector
    detector = StrawberryDetector(model_path=args.model, use_gpu=args.use_gpu)

    # 3. Load Image
    print(f"\nProcessing image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Could not read the image file. It might be corrupted or in an unsupported format.")
        return

    # 4. Run Inference
    d_bboxes, d_scores, d_class_ids = detector(image)
    print("\n--- Detection Results ---")
    if len(d_bboxes) > 0:
        for i in range(len(d_bboxes)):
            print(f"Class: {CLASS_NAMES.get(d_class_ids[i], 'Unknown')}, Score: {d_scores[i]:.2f}, BBox: {d_bboxes[i].astype(int)}")
    else:
        print("No objects detected.")

    # 5. Draw Results
    labeled_img = draw_strawberry_inferences(image.copy(), d_bboxes, d_scores, d_class_ids)

    # 6. Save and Display the Output Image
    # Construct a new filename for the output
    input_path = Path(args.image)
    output_filename = f"inferred_{input_path.name}"
    cv2.imwrite(output_filename, labeled_img)
    print(f"\n✅ Successfully saved labeled image to: {output_filename}")

    # Display the image in a pop-up window
    cv2.imshow('Detection Result', labeled_img)
    print("Press any key to close the image window...")
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
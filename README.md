🍓 Strawberry-MOT: A Framework for Strawberry Detection, Ripeness Classification, and Tracking

This repository provides a complete pipeline for detecting strawberries in video streams, classifying their ripeness stage (unripe, semi-ripe, ripe), and tracking them using multiple state-of-the-art multi-object tracking (MOT) algorithms to support accurate yield estimation.
🔑 Key Features
🍓 High-Performance Detector
Custom-trained YOLOv8s model (c3x + head + WIoU) (custom_models>best.onxx) for precise strawberry detection.

🎯 3-Class Ripeness Classification
Classifies detected strawberries as unripe, semi-ripe, or ripe.

🔄 Modular Tracking Framework
Easily switch between multiple MOT algorithms: mc_bytetrack, mc_sort, mc_norfair, mc_motpy and mc_deepsort. 

DeepSORT requires older versions of TensorFlow and CUDA. Create a separate environment (deepsort_env) when using it.

See requirements.txt and requirements_deepsort for details.

🖼️ Single Image Detection
Run the detector on a single image using:

python test_detector.py --image "C:\path\to\your\image.jpg"

🎥 Video Tracking & Counting
Main script for video processing:

python main.py --input <path_to_video_or_folder> --tracker <tracker_name>

Pretrained weights for YOLOv3-tiny, YOLOv5, YOLOv6, and YOLOv8s are included in the Detector/ folder.

📦 Dataset Access
The dataset used for detection, classification, and tracking is currently not publicly available.
Researchers interested in accessing the data can contact us via this repository’s issue tracker.




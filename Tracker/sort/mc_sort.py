# File: TGI/Tracker/mc_sort.py

import numpy as np
from .sort import Sort # Import the Sort class

class MCSort:
    def __init__(self, **kwargs):
        """
        Initializes the SORT tracker.
        max_age: Maximum number of frames to keep a track alive without new detections.
        min_hits: Minimum number of associated detections before a track is initialized.
        """
        self.sort_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    def __call__(self, frame, detections, scores, class_ids):
        """
        Updates the tracker with new detections.
        
        Args:
            frame: The current video frame (not used by SORT, but passed for API consistency).
            detections: A list or numpy array of bounding boxes in [x1, y1, x2, y2] format.
            scores: A list or numpy array of detection confidence scores.
            class_ids: A list or numpy array of class IDs.

        Returns:
            A tuple of (track_ids, bboxes, scores, class_ids) for the active tracks.
        """
        if detections is None or len(detections) == 0:
            # SORT's update function requires at least an empty array
            tracked_objects = self.sort_tracker.update(np.empty((0, 5)))
            return [], [], [], []
        
        # SORT expects detections in the format [x1, y1, x2, y2, score]
        detections_for_sort = np.hstack((detections, scores[:, np.newaxis]))

        # Update the tracker
        tracked_objects = self.sort_tracker.update(detections_for_sort)

        # Process the tracked objects
        track_ids = []
        bboxes = []
        
        if len(tracked_objects) > 0:
            # tracked_objects format is [x1, y1, x2, y2, track_id]
            bboxes = tracked_objects[:, :4]
            track_ids = tracked_objects[:, 4].astype(int)

        # Note: SORT does not track scores or class_ids. We return empty lists
        # for these to maintain API consistency, or you could re-associate them.
        # For simplicity, we return empty arrays for scores and class_ids.
        final_scores = [1.0] * len(track_ids) # Or re-associate if needed
        final_class_ids = [0] * len(track_ids) # Or re-associate if needed

        return track_ids, bboxes, final_scores, final_class_ids
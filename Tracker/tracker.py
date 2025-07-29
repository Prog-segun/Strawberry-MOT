import json

class MultiObjectTracker(object):
    def __init__(
        self,
        tracker_name='motpy',
        fps=30,
        use_gpu=False,
    ):
        self.fps = round(fps, 2)
        self.tracker_name = tracker_name
        self.tracker = None
        self.config = None
        self.use_gpu = use_gpu

        if self.tracker_name == 'motpy':
            from .motpy.motpy import Motpy
            self.use_gpu = False
            # ... (rest of motpy code)

        elif self.tracker_name == 'mc_bytetrack':
            from .bytetrack.mc_bytetrack import MultiClassByteTrack
            self.use_gpu = False
            with open('Tracker/bytetrack/config.json') as fp:
                self.config = json.load(fp)
            if self.config is not None:
                self.tracker = MultiClassByteTrack(
                    fps=self.fps,  # <-- THE FIX IS HERE
                    track_thresh=self.config['track_thresh'],
                    track_buffer=self.config['track_buffer'],
                    match_thresh=self.config['match_thresh'],
                    min_box_area=self.config['min_box_area'],
                    mot20=self.config['mot20'],
                )

        elif self.tracker_name == 'mc_norfair':
            from .norfair.mc_norfair import MultiClassNorfair
            self.use_gpu = False
            with open('Tracker/norfair/config.json') as fp:
                self.config = json.load(fp)
            if self.config is not None:
                self.tracker = MultiClassNorfair(
                    fps=self.fps,
                    max_distance_between_points=self.
                    config['max_distance_between_points'],
                )
        
        elif self.tracker_name == 'mc_sort':
            from .sort.mc_sort import MCSort
            self.use_gpu = False 
            with open('Tracker/sort/config.json') as fp:
                self.config = json.load(fp)
            if self.config is not None:
                self.tracker = MCSort(
                    max_age=self.config['max_age'],
                    min_hits=self.config['min_hits'],
                    iou_threshold=self.config['iou_threshold'],
                )

        elif self.tracker_name == 'mc_deepsort':
            from .deep_sort.mc_deepsort import MCDeepSort
            self.config = {'note': 'DeepSORT config is handled by its internal wrapper.'}
            self.tracker = MCDeepSort(use_gpu=self.use_gpu)
            
        else:
            raise ValueError('Invalid Tracker Name')

    def __call__(self, image, bboxes, scores, class_ids):
        if self.tracker is not None:
            results = self.tracker(image, bboxes, scores, class_ids)
        else:
            raise ValueError('Tracker is None')
        return results[0], results[1], results[2], results[3]

    def print_info(self):
        from pprint import pprint
        print('Tracker:', self.tracker_name)
        print('FPS:', self.fps)
        print('GPU:', self.use_gpu)
        pprint(self.config, indent=4)
        print()
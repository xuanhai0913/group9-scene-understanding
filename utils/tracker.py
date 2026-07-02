import numpy as np

class Tracker:
    """A lightweight multi-object tracker with a linear motion model (velocity projection)
    and class-specific occlusion lifetimes.

    It associates detections in the current frame with tracks from previous frames,
    preserving vehicle/pedestrian/traffic-light IDs as they move or get temporarily occluded.
    """

    def __init__(self, min_iou=0.15, alpha=0.6, decay=0.9):
        self.next_id = 0
        self.tracks = {}  # tid -> track dict
        self.min_iou = min_iou
        self.alpha = alpha      # Smoothing factor for velocity update (exponential moving average)
        self.decay = decay      # Velocity decay factor when unmatched

    def _get_max_age(self, obj_type):
        """Stationary objects like traffic lights or stop signs should have a much
        longer lifetime during occlusion than moving objects.
        """
        t = obj_type.lower()
        if 'light' in t or 'sign' in t:
            return 90  # Keep alive for ~3 seconds at 30 fps (great for occluded traffic lights)
        elif 'vehicle' in t or 'car' in t or 'truck' in t or 'bus' in t:
            return 30  # Keep alive for ~1 second at 30 fps (great for vehicle occlusion)
        else:
            return 20  # Default for humans, etc.

    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def _compute_centroid_distance(self, box1, box2):
        """Compute the Euclidean distance between centroids of two boxes."""
        c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def update(self, detections):
        """Update tracks with the current frame detections.

        Parameters
        ----------
        detections: list of dict
            Each dict must contain ``box`` (tuple of (xmin, ymin, xmax, ymax)),
            ``depth`` (float) and ``type`` (string) keys – exactly as produced by
            the detection logic in ``main.py``.

        Returns
        -------
        dict
            Mapping from track id to a track dictionary with the keys accessed
            in ``main.py``: ``box``, ``type``, ``depth_history``, ``delta_d``,
            ``dist``, ``is_in_lane``, ``box_history``.
        """
        # Step 1: Predict position of all existing tracks for the current frame
        for tid, track in self.tracks.items():
            track['age'] += 1
            # Predict box position based on current velocity
            box = track['box']
            vel = track['velocity']
            predicted_box = (
                box[0] + vel[0],
                box[1] + vel[1],
                box[2] + vel[2],
                box[3] + vel[3]
            )
            track['box'] = predicted_box
            # Decay the velocity slightly during occlusion
            if track['age'] > 1:
                track['velocity'] = [v * self.decay for v in vel]

        # Step 2: Match current detections with existing tracks
        matched_detections = [False] * len(detections)
        matched_tracks = set()

        # Candidates will store (match_score, tid, det_idx)
        # We prioritize IoU. If IoU is low/zero, we can use a distance fallback.
        candidates = []
        for tid, track in self.tracks.items():
            max_age = self._get_max_age(track['type'])
            if track['age'] > max_age:
                continue

            for det_idx, det in enumerate(detections):
                iou = self._compute_iou(track['box'], det['box'])
                
                # Calculate centroid distance as fallback
                dist = self._compute_centroid_distance(track['box'], det['box'])
                
                # Determine normalized centroid distance (normalized by track box diagonal or width/height)
                track_w = max(1.0, track['box'][2] - track['box'][0])
                track_h = max(1.0, track['box'][3] - track['box'][1])
                diag = np.sqrt(track_w**2 + track_h**2)
                norm_dist = dist / diag
                
                # Match score formula:
                # We want to favor IoU. If IoU >= min_iou, score is 1.0 + IoU
                # If IoU is lower but distance is very small (norm_dist < 0.45), we can use distance match
                if iou >= self.min_iou:
                    score = 1.0 + iou
                    candidates.append((score, tid, det_idx))
                elif norm_dist < 0.45:  # Centroid is very close
                    score = 0.5 * (1.0 - norm_dist)  # Between 0.275 and 0.5
                    candidates.append((score, tid, det_idx))

        # Sort candidate matches by score in descending order
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Greedy matching
        for score, tid, det_idx in candidates:
            if tid in matched_tracks or matched_detections[det_idx]:
                continue
            
            # Match found!
            matched_tracks.add(tid)
            matched_detections[det_idx] = True

            det = detections[det_idx]
            track = self.tracks[tid]
            
            # Update velocity before updating the bounding box
            old_actual = track.get('last_actual_box')
            new_box = det.get('box')
            
            # Calculate instant velocity based on the frame interval since last actual detection
            dt = track['age']
            if old_actual is not None and dt > 0:
                instant_vel = [(new_box[i] - old_actual[i]) / dt for i in range(4)]
            else:
                instant_vel = [0.0, 0.0, 0.0, 0.0]
            
            # If the track was just created, velocity is 0. Else, we smooth it.
            if all(v == 0.0 for v in track['velocity']):
                track['velocity'] = instant_vel
            else:
                track['velocity'] = [
                    self.alpha * instant_vel[i] + (1 - self.alpha) * track['velocity'][i]
                    for i in range(4)
                ]

            # Update track parameters
            track['box'] = new_box
            track['last_actual_box'] = new_box
            track['type'] = det.get('type', 'unknown')
            
            depth = det.get('depth', 0.0)
            track['depth_history'].append(depth)
            if len(track['depth_history']) > 30:
                track['depth_history'].pop(0)

            if len(track['depth_history']) >= 2:
                track['delta_d'] = track['depth_history'][-1] - track['depth_history'][-2]
            else:
                track['delta_d'] = 0.0

            track['box_history'].append(new_box)
            if len(track['box_history']) > 30:
                track['box_history'].pop(0)

            track['age'] = 0  # Reset age since it was matched in this frame

        # Step 3: Create new tracks for unmatched detections
        for det_idx, matched in enumerate(matched_detections):
            if not matched:
                det = detections[det_idx]
                box = det.get('box')
                depth = det.get('depth', 0.0)
                
                new_track = {
                    'box': box,
                    'last_actual_box': box,
                    'type': det.get('type', 'unknown'),
                    'depth_history': [depth],
                    'delta_d': 0.0,
                    'dist': 0.0,
                    'is_in_lane': True,
                    'box_history': [box],
                    'age': 0,
                    'velocity': [0.0, 0.0, 0.0, 0.0]
                }
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Step 4: Remove tracks that exceeded their class-specific max age
        expired_tids = []
        for tid, track in self.tracks.items():
            max_age = self._get_max_age(track['type'])
            if track['age'] > max_age:
                expired_tids.append(tid)
        for tid in expired_tids:
            del self.tracks[tid]

        # Step 5: Return active tracks and temporarily lost/predicted tracks (age <= 2) for anti-flicker rendering
        current_active_tracks = {
            tid: track for tid, track in self.tracks.items()
            if track['age'] <= 2
        }

        return current_active_tracks

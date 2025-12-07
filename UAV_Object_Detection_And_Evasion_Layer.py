import sys
import numpy as np
import math
from ultralytics import YOLO
import cv2
import time
import json

class VideoReader():
    def __init__(self):
        self.capture = None
        self.max_width=1920 
        self.max_height=1080 
        self.target_fps=10
        self.auto_resize=True

        self.frame_width = None 
        self.frame_height = None 
        self.video_fps = None
        self.frame_index = 0

    def load_video(self, video_path):
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        # Video properties
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS) or 30

        # Resolution check
        if self.frame_width > self.max_width or self.frame_height > self.max_height:
            if not self.auto_resize:
                raise ValueError(
                    f"Video resolution too large ({self.frame_width}x{self.frame_height}). "
                    f"Max allowed is {self.max_width}x{self.max_height}")

    def adjust_frame_rate(self):
        """
        Adaptive frame skipping to match target FPS.
        """
        now = time.time()
        elapsed = now - self.last_frame_time

        # If processing too fast, wait
        required_delay = 1.0 / self.target_fps

        if elapsed < required_delay:
            time.sleep(required_delay - elapsed)

        self.last_frame_time = time.time()

    def _resize_to_max(self, frame):
        h, w = frame.shape[:2]

        scale = min(self.max_width / w, self.max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def read_frame(self):
        if self.capture is None:
            raise RuntimeError("Video not loaded. Call load_video() first.")

        self.adjust_frame_rate()

        ok, frame = self.capture.read()
        if not ok:
            return None  # End of video or unreadable frame

        # Validate frame integrity
        if frame is None or frame.size == 0:
            return None

        # Resize oversized frames
        h, w = frame.shape[:2]
        if w > self.max_width or h > self.max_height:
            if self.auto_resize:
                frame = self._resize_to_max(frame)
            else:
                raise ValueError("Frame exceeds maximum allowed size.")
        self.frame_index += 1

        return frame

class VideoFrameAnalyser():
    def __init__(self, image_path=None):
        self.fov = 100
        self.screen_width = None
        self.screen_height = None

        if image_path:
            self.load_and_set_properties(image_path)

    def load_video_frame(self, frame):
        if frame is None:
            raise FileNotFoundError(f"Could not load image: ")
        self.screen_height, self.screen_width = frame.shape[:2]

    def load_and_set_properties(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        # Width & Height directly from image
        self.screen_height, self.screen_width = img.shape[:2]

        return img

    def load_image(self, path):
        """Load normal BGR image for YOLO."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img

    def image_greyscale(self, path):
        """Loads image and converts to grayscale."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def get_fov(self):
        return self.fov

    def get_screen_width(self):
        return self.screen_width 

    def get_screen_height(self):
        return self.screen_height

class Obstacle:
    def __init__(self, bbox, conf, class_name):
        self.bbox = bbox
        self.conf = conf
        self.class_name = class_name

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1] 

    def centre(self):
        cx = (self.bbox[0] + self.bbox[2]) // 2
        cy = (self.bbox[1] + self.bbox[3]) // 2
        return (cx, cy)

    def estimated_distance(self, frame_height, k):
        #k is assumed camera constant -> use k = 0.8
        return (frame_height * k) / self.height

    def horizontal_offset(self, frame_width):
        #This gives an offset from -1 to +1
        centre = self.centre()
        cx  = centre[0]
        w = frame_width / 2
        return (cx - w) / w

    def top_left(self):
        return (self.bbox[0], self.bbox[1])

    def bottom_left(self):
        return (self.bbox[0], self.bbox[3])

    def top_right(self):
        return (self.bbox[2], self.bbox[1])

    def bottom_right(self):
        return (self.bbox[2], self.bbox[3])

    def bounding_box(self):
        return self.bbox

class EmptyBox():
    def __init__(self, points):
        self.points = points 

    def centre(self):
        cx = (self.points[0] + self.points[2]) // 2
        cy = (self.points[1] + self.points[3]) // 2
        return (cx, cy)

    def width(self):
        return self.points[2] - self.points[0] 

    def height(self):
        return self.points[3] - self.points[1]

    def clearance_score(self, frame_height):
        #calculates clearance, or how much space there is relative to that of the frame
        height = self.height()
        return height / frame_height

    def deviation_score(self, trajectory, screen_width, screen_height):
        cx, cy = self.centre()
        vx = cx - (screen_width) / 2
        vy = cy - (screen_height) / 2

        dot = vx * trajectory[0] + vy * trajectory[1]
        len_u = math.sqrt(vx*vx + vy*vy)
        len_v = math.sqrt(trajectory[0]**2 + trajectory[1]**2)

        if len_u == 0 or len_v == 0:
            return 0.0

        cos0 = dot / (len_u * len_v)
        cos0 = max(-1, min(1, cos0))

        return (1 - cos0) / 2


class YOLOObstacleDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect_shapes(self, frame):
        # Ensure frame is a valid uint8 image
        threshold = 150
        min_area = 180
        frame = np.asarray(frame)
        if frame is None:
            raise ValueError("detect_shapes() received None frame")
        # Convert to grayscale safely
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:
            gray = frame.copy()
        else:
            raise ValueError(f"Invalid frame shape: {frame.shape}")
        # Ensure grayscale is uint8
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            gray = gray.astype(np.uint8)
        # Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Threshold (invert black shapes → white)
        ok, thresh = cv2.threshold(
            blur, threshold, 255, cv2.THRESH_BINARY_INV
        )
        if not ok or thresh is None:
            raise RuntimeError("Thresholding failed – output is None")
        # Ensure thresh is binary uint8
        thresh = thresh.astype(np.uint8)
        # ---- NOW SAFE ----
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        obstacles = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
        x, y, w, h = cv2.boundingRect(c)

        obstacles.append(
            Obstacle([x, y, x + w, y + h], 1.0, "contour-shape")
        )
        return obstacles

    def detect_real_world_obstacles(self, frame):
        results = self.model.predict(frame, verbose=False)

        obstacles = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                obstacles.append (
                    Obstacle([x1, y1, x2, y2], conf, cls.__str__())
                )

        return obstacles

    def detect(self, frame):
        #Uses detection based on both object recognition and shape recognition
        shapes = self.detect_shapes(frame)
        obstacles = self.detect_real_world_obstacles(frame)
        final_list_obstacles = shapes
        for obs in obstacles:
            final_list_obstacles.append(obs)
        return final_list_obstacles

class Vector():
    
    @staticmethod
    def project_vector_3d_to_2d(v, fov, w, h):
        v = v / np.linalg.norm(v)
        if v[2] <= 0: 
            v[2] = 1e-3  # prevent blow-up / behind-camera case
        f = (w/2) / np.tan(np.radians(fov/2))

        x = (v[0] / v[2]) * f + w/2
        y = (v[1] / v[2]) * f + h/2
        return [x, y]

    @staticmethod
    def convert_vector_to_3D(fov, vect, w, h):
        x, y = vect

        # Normalize to [-1,1]
        x_norm = (2 * x / w) - 1
        y_norm = 1 - (2 * y / h)

        fov_rad = math.radians(fov)
        scale = math.tan(fov_rad / 2)

        x_c = x_norm * scale
        y_c = y_norm * scale
        z_c = 1

        mag = math.sqrt(x_c*x_c + y_c*y_c + z_c*z_c)
        return [x_c/mag, y_c/mag, z_c/mag]

class DroneState():
    def __init__(self, position, velocity, dt = 0.1):
        self.pos = position
        self.vel = velocity
        self.dt = dt
        self.speed = 10

    def new_position(self):
        x = self.pos[0] + self.vel[0]
        y = self.pos[1] + self.vel[1] 
        z = self.pos[2] + self.vel[2] 
        return (x, y, z)

    def rad(self, deg):
        return math.radians(deg)

    def get_trajectory(self):
        return self.vel

    def set_trajectory(self, trajectory):
        self.vel = trajectory

    def update_trajectory(self, new_trajectory):
        """
        Update the drone's velocity vector and position based on the new trajectory.

        new_trajectory: 3-element list or tuple representing a direction vector in 3D.
        """

        # --- 1. Normalise the incoming vector ---
        nx, ny, nz = new_trajectory
        mag = math.sqrt(nx*nx + ny*ny + nz*nz)

        # Avoid division by zero
        if mag == 0:
            return  # Keep current velocity and do nothing

        nx /= mag
        ny /= mag
        nz /= mag

        # --- 2. Update velocity ---
        # Velocity = direction x speed
        self.vel = [nx * self.speed, ny * self.speed, nz * self.speed]

        # --- 3. Update position ---
        # Standard physics: x_new = x_old + v * dt
        self.pos[0] += self.vel[0] * self.dt
        self.pos[1] += self.vel[1] * self.dt
        self.pos[2] += self.vel[2] * self.dt

    def alter_z(self, increment):
        self.vel[2] += increment

class Avoider():
    def __init__(self, drone, obstacles, frame_width, frame_height, fov, weight_alignment = 0.4, weight_clearance = 0.6):
        self.dist_threshold = 30
        self.drone = drone 
        self.obstacles = obstacles 
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fov = fov

        self.weight_alignment = weight_alignment
        self.weight_clearance = weight_clearance
        self.learning_rate = 0.02

        self.optimal_trajectory = None

    def set_hyperparameters(self, hyperparameters_dict):
        self.weight_alignment = hyperparameters_dict.get("weight_alignment")
        self.weight_clearance = hyperparameters_dict.get("weight_clearance")

    def new_position_2D(self):
        point = self.drone.new_position()
        return Vector.project_vector_3d_to_2d(point, self.fov, self.frame_width, self.frame_height)

    def obstacle_in_path(self, obstacle, frame_width, frame_height):
        k = 0.8
        dist = obstacle.estimated_distance(frame_height, k)
        off = obstacle.horizontal_offset(frame_width)
        return (dist < self.dist_threshold and abs(off) < 0.3)

    def convert_vector_to_3D(self, vect, w, h):
        x, y = vect

        # Normalize to [-1,1]
        x_norm = (2 * x / w) - 1
        y_norm = 1 - (2 * y / h)

        fov_rad = math.radians(self.fov)
        scale = math.tan(fov_rad / 2)

        x_c = x_norm * scale
        y_c = y_norm * scale
        z_c = 1

        mag = math.sqrt(x_c*x_c + y_c*y_c + z_c*z_c)
        return [x_c/mag, y_c/mag, z_c/mag]

    def get_empty_spaces(self):
        empty_spaces = []
        # sort obstacles by x1
        obstacles_sorted = sorted(self.obstacles, key=lambda o: o.bbox[0])

        last_right = 0

        for obs in obstacles_sorted:
            x1, y1, x2, y2 = obs.bbox
            # empty region between last obstacle and this one
            if x1 > last_right:
                empty_spaces.append(EmptyBox((last_right, 0, x1, self.frame_height)))

            last_right = max(last_right, x2)
        # empty region to the right of the last obstacle
        if last_right < self.frame_width:
            empty_spaces.append(EmptyBox((last_right, 0, self.frame_width, self.frame_height)))
        return empty_spaces

    def max_distance(self):
        dist = 0
        max_dist = 0
        for obstacle in self.obstacles:
            dist = obstacle.estimated_distance(self.frame_height, 0.8)
            if dist > max_dist:
                max_dist = dist 
        return max_dist

    def update_hyperparameters(self, best):
        print(best)
        c = best["clearance"]
        d = best["deviation"]

        # Normalize ratios
        total = c + d + 1e-6
        c_ratio = c / total
        d_ratio = d / total

        # Move weights toward c_ratio/d_ratio
        self.weight_clearance += self.learning_rate * (c_ratio - self.weight_clearance)
        self.weight_alignment += self.learning_rate * (d_ratio - self.weight_alignment)

        # Re-normalise
        s = self.weight_clearance + self.weight_alignment
        self.weight_clearance /= s
        self.weight_alignment /= s

    def calculate_optimal_trajectory(self):
        point_following_trajectory = self.new_position_2D()
        empty_spaces = self.get_empty_spaces()
        clearance_score = 0
        deviation_score = 0
        max_total = 0

        best_scored_point = None
        for space in empty_spaces:
            clearance_score = space.clearance_score(self.frame_height)
            deviation_score = space.deviation_score(point_following_trajectory, self.frame_width, self.frame_height)
            total_score =  deviation_score * self.weight_alignment + clearance_score * self.weight_clearance
            if total_score > max_total:
                max_total = total_score
                best_scored_point = {"clearance": clearance_score, "deviation": deviation_score, "total": total_score, "coordinates": space.centre()}

        if best_scored_point is not None:
            cx, cy = best_scored_point["coordinates"]
            new_trajectory = self.convert_vector_to_3D((cx, cy), self.frame_width, self.frame_height)
            self.optimal_trajectory = new_trajectory
            self.update_hyperparameters(best_scored_point)

        else:
            if len(self.obstacles) > 0:
                #if there are no good points, increase z coordinate (height)
                d = self.max_distance()
                self.drone.alter_z(d)
                self.optimal_trajectory = self.drone.get_trajectory()
            else:
                #if there are no obstacles, let the trajectory remain the same
                self.optimal_trajectory = self.drone.get_trajectory()

    def set_optimal_trajectory(self):
        if self.optimal_trajectory is not None:
            self.drone.update_trajectory(self.optimal_trajectory)

    def get_optimal_trajectory(self):
        return self.optimal_trajectory

    def print_hyperparameters(self):
        print(f"clearance weight: {self.weight_clearance}, alignment weight: {self.weight_alignment}, learning rate: {self.learning_rate}")

    def get_hyperparameters(self):
        return f"clearance weight: {self.weight_clearance}, alignment weight: {self.weight_alignment}, learning rate: {self.learning_rate}"

    def hyperparameters_dict(self):
        return {"weight_clearance": self.weight_clearance, "weight_alignment":self.weight_alignment}

    @staticmethod
    def from_dict(cls, data):
        return cls(**data)

    def save(self):
        with open("hyperparameters.txt", "w") as f:
            json.dump(self.hyperparameters_dict(), f, indent = 4)

    def load(self):
        with open("hyperparameters.txt", "r") as f:
            data = json.load(f)
            return Avoider.from_dict(data)

class Runner():
    def __init__(self):
        pass 

    def process_video(self, video_path, initial_vector_3d):
        reader = VideoReader()
        reader.load_video(video_path)

        frame_analyser = VideoFrameAnalyser()
        detector = YOLOObstacleDetector()
        drone = DroneState([0,0,0], initial_vector_3d)

        frame = reader.progress_frame()
        while frame is not None:
            # Initialize analyser after getting first frame
            frame_analyser.load_video_frame(frame)

            obstacles = detector.detect(frame)
            screen_width = frame_analyser.get_screen_width()
            screen_height = frame_analyser.get_screen_height()
            fov = frame_analyser.get_fov()

            avoider = Avoider(drone, obstacles, screen_width, screen_height, fov)
            hyperparameters = avoider.load()
            avoider.set_hyperparameters(hyperparameters)

            avoider.calculate_optimal_trajectory()
            avoider.set_optimal_trajectory

            results =  {"frame_index": reader.frame_index, "previous trajectory": initial_vector_3d,"current trajectory": avoider.get_optimal_trajectory(), "obstacle_count": len(obstacles), "hyperparameters": avoider.get_hyperparameters}

            initial_vector_3d = avoider.get_optimal_trajectory()

            print(results)
            print("")

    def run_image_test(self, img_path, current_vector_3d):
        """This is a high level method that loads all the components of the system"""
        #Starting video frame analysis
        frame_analyser = VideoFrameAnalyser(img_path)
        #Starting obstacle detection
        detector = YOLOObstacleDetector()
        # Starting drone state
        current_position = [0.0, 0.0, 0.0]
        drone = DroneState(current_position, current_vector_3d)
        # Screen + camera info
        screen_width = frame_analyser.get_screen_width()
        screen_height = frame_analyser.get_screen_height()
        fov = frame_analyser.get_fov()
        #getting obstacles
        frame = frame_analyser.load_image(img_path)
        obstacles = detector.detect(frame)
        print(f"obstacles detected {len(obstacles)}")
        # Visualise the obstacles on the frame
        for obs in obstacles:
            cv2.rectangle(frame, 
                      (obs.bbox[0], obs.bbox[1]),
                      (obs.bbox[2], obs.bbox[3]),
                      (255, 0, 0), 2)
        #Starting evasion system
        avoider = Avoider(drone, obstacles, screen_width, screen_height, fov) 
        # === Calculate actual optimal trajectory ===
        avoider.calculate_optimal_trajectory()
        # Apply new trajectory to the drone
        avoider.set_optimal_trajectory()
        # Retrieve the new trajectory
        new_traj = avoider.get_optimal_trajectory()
        print("Original 3D trajectory:", current_vector_3d)
        print("Optimal 3D trajectory:", new_traj)
        avoider.print_hyperparameters()
        #show 2d vectors of previous and current trajectories
        cx = screen_width // 2
        cy = screen_height // 2
        # Draw previous vector from screen center
        prev_traj = Vector.project_vector_3d_to_2d(current_vector_3d, fov, screen_width, screen_height)
        new_traj = Vector.project_vector_3d_to_2d(new_traj, fov, screen_width, screen_height)
        print("trajectories projected in 2D:")
        print(f"prev_traj {prev_traj}, new_traj {new_traj}")
        # Draw previous vector
        cv2.line(frame,
            (cx, cy),
            (int(prev_traj[0]), int(prev_traj[1])),
            (0, 0, 0))
        # Draw new vector
        cv2.line(frame,
            (cx, cy),
            (int(new_traj[0]), int(new_traj[1])),
            (0, 255, 0))
        # Show test frame
        cv2.namedWindow("Test Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Test Frame", frame)
        cv2.resizeWindow("Test Frame", 1280, 720)
        cv2.waitKey(0)

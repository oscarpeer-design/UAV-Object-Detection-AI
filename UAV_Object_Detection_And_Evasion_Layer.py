import sys
import numpy as np
import math
from ultralytics import YOLO
import cv2

class VideoFrameAnalyser():
    def __init__(self, image_path=None):
        self.fov = 100
        self.screen_width = None
        self.screen_height = None

        if image_path:
            self.load_and_set_properties(image_path)

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

    def detect(self, frame):
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

class DroneState:
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

    def convert_velocity_2D(self, fov_deg):
        fov_rad = math.radians(fov_deg)
        f = 1 / math.tan(fov_rad / 2)
        if self.vel[2] == 0:
            return [0, 0]
        x_screen = (self.vel[0] / self.vel[2]) * f
        y_screen = (self.vel[1] / self.vel[2]) * f
        return [x_screen, y_screen]

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
    def __init__(self, drone, obstacles, frame_width, frame_height, fov):
        self.dist_threshold = 30
        self.drone = drone 
        self.obstacles = obstacles 
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fov = fov

        self.weight_alignment = 0.4
        self.weight_clearance = self.complement(self.weight_alignment)
        self.learning_rate = 0.02

        self.optimal_trajectory = None

    def complement(self, p):
        return 1.0 - p

    def convert_point_2D(self, point):
        fov_rad = math.radians(self.fov)
        f = 1 / math.tan(fov_rad / 2)

        X, Y, Z = point[0], point[1], point[2]
        if Z == 0:
            Z = 0.0001

        x_screen = (X / Z) * f
        y_screen = (Y / Z) * f
        return [x_screen, y_screen]

    def new_position_2D(self):
        point = self.drone.new_position()
        return self.convert_point_2D(point)

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

    def optimal_trajectory(self):
        return self.optimal_trajectory

    def print_hyperparameters(self):
        print(f"clearance weight: {self.weight_clearance}, alignment weight: {self.weight_alignment}, learning rate: {self.learning_rate}")

class Runner():
    def __init__(self):
        pass 

    def analyse_optimal_trajectory(self, img_path):
        """This is a high level method that loads all the components of the system"""
        #Starting video frame analysis
        frame_analyser = VideoFrameAnalyser(img_path)
        #Starting obstacle detection
        detector = YOLOObstacleDetector()
        # Starting drone state
        current_vector_3d = np.array([1.0, 0.2, -0.1])
        current_position = [0.0, 0.0, 0.0]
        drone = DroneState(current_position, current_vector_3d)
        # Screen + camera info
        screen_width = frame_analyser.get_screen_width()
        screen_height = frame_analyser.get_screen_height()
        fov = frame_analyser.get_fov()
        #getting obstacles
        frame = frame_analyser.load_image(img_path)
        obstacles = detector.detect(frame)
        #Starting evasion system
        avoider = Avoider(drone, obstacles, screen_width, screen_height, fov)
        # === Calculate actual optimal trajectory ===
        avoider.calculate_optimal_trajectory()
        # Apply new trajectory to the drone
        avoider.set_optimal_trajectory()
        # Retrieve the new trajectory
        new_vector = drone.get_trajectory()
        print("Original 3D vector:", current_vector_3d)
        print("Adjusted 3D vector:", new_vector)
        avoider.print_hyperparameters()
        # Show debug frame
        cv2.imshow("Test Frame", frame)
        cv2.waitKey(0)

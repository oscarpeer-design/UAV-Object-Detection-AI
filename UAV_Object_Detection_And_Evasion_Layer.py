import sys
import numpy
import math
from ultralytics import YOLO
import cv2

class VideoFrameAnalyser():
    def __init__(self):
        self.fov = 100 #stub
        self.screen_height = 200
        self.screen_width = 200

    def get_fov(self):
        return self.fov

    def get_screen_width(self):
        return self.screen_width 

    def get_screen_height(self):
        return self.screen_height

class Obstacle():
    def __init__(self, bbox, conf, class_name):
        self.bbox = bbox
        self.conf = conf 
        self.class_name = class_name 

        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]

    def width(self):
        return self.width 

    def height(self):
        return self.height 

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

    def bottom_left(self):
        x = self.bbox[0]
        y = self.bbox[1] - self.height 
        return (x, y)

    def top_left(self):
        x = self.bbox[0]
        y = self.bbox[1] 
        return (x, y)

    def top_right(self):
        x = self.bbox[2] 
        y = self.bbox[3]
        return (x, y)

    def bottom_right(self):
        x = self.bbox[2] 
        y = self.bbox[3] - self.height 
        return (x, y)

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

    def deviation_score(self, trajectory):
        #This calculates the deviation of one vector from another, between 0 and 1
        # u.v = |u||v|cos(theta)
        # Therefore cos(theta) = u.v / (|u||v|)
        # This is normalised between -1 and 1
        #-1 means 100%, 0 means 50% deviation, 1 means 0% deviation
        centre = self.centre()
        dot_product = centre[0] * trajectory[0] + centre[1] * trajectory[1]
        len_u = math.sqrt(centre[0] ** 2 + centre[1] ** 2)
        len_v = math.sqrt(trajectory[0] ** 2 + trajectory[1] ** 2)

        costheta = dot_product / (len_u * len_v)
        #clamp angle between -1 and 1
        costheta = max(-1, min(1, costheta))

        deviation_score = (1.0 - costheta) / 2.0 
        return deviation_score

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
    def __init__(self, position, velocity):
        self.pos = position
        self.vel = velocity

    def new_position(self):
        x = self.pos[0] + self.vel[0]
        y = self.pos[1] + self.vel[1] 
        z = self.pos[2] + self.vel[2] 
        return (x, y, z)

    def convert_velocity_2D(self, fov):
        f = 1 / math.tan(fov / 2)
        x_screen = (self.vel[0] / self.vel[2]) * f
        y_screen = (self.vel[1] / self.vel[2]) * f
        return [x_screen, y_screen]

    def get_trajectory(self):
        return self.vel

    def set_trajectory(self, trajectory):
        self.vel = trajectory

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

        self.weight_direction = 0.4
        self.weight_clearance = self.complement(self.weight_direction)

        self.optimal_trajectory = None

    def complement(self, p):
        return 1.0 - p

    def convert_point_2D(self, point):
        f = 1 / math.tan(self.fov / 2)
        x_screen = (point[0] / point[2]) * f
        y_screen = (point[1] / point[2]) * f
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
        x_norm = (2 * vect[0]) / w -1
        y_norm = 1 - (2 * vect[1]) / h

        x_c = x_norm * math.tan(self.fov / 2)
        y_c = y_norm * math.tan(self.fov / 2)
        z_c = 1

        d = math.sqrt(x_c ** 2 + y_c ** 2 + z_c ** 2)
        x = x_c / d 
        y = y_c / d 
        z = z_c / d
        return [x, y, z]

    def get_empty_spaces(self):
        empty_spaces = []
        start_x = 0
        start_y = 0
        for obstacle in self.obstacles:
            bbox = obstacle.bounding_box()
            x0 = bbox[0] - start_x 
            y0 = bbox[1] - start_y 
            x1 = bbox[2] - start_x 
            y1 = bbox[3] - start_y 
            width = x1 - x0 
            height = y1 - y0 

            if width >= self.dist_threshold and height >= self.dist_threshold:
                points = (x0, y0, x1, y1) 
                empty_spaces.append(EmptyBox(points))

            start_x = bbox[2]
            start_y = bbox[1]

        return empty_spaces

    def max_distance(self):
        dist = 0
        max_dist = 0
        for obstacle in self.obstacles:
            dist = obstacle.estimated_distance(self.frame_height, 0.8)
            if dist > max_dist:
                max_dist = dist 
        return max_dist

    def update_hyperparameters(self, best_scored_point):
        pass 

    def calculate_optimal_trajectory(self):
        point_following_trajectory = self.new_position_2D()
        empty_spaces = self.get_empty_spaces()
        clearance_score = 0
        deviation_score = 0
        max_total = 0

        best_scored_point = None
        for space in empty_spaces:
            clearance_score = space.clearance_score(self.frame_height)
            deviation_score = space.deviation_score(point_following_trajectory)
            total_score =  deviation_score * self.weight_direction + clearance_score * self.weight_clearance
            if total_score > max_total:
                max_total = total_score
                best_scored_point = {"clearance": clearance_score, "deviation": deviation_score, "total": total_score, "coordinates": space.centre()}

        if best_scored_point is not None:
            coords = best_scored_point.get("coordinates")
            new_trajectory = self.convert_vector_to_3D(coords, self.frame_width, self.frame_height)
            self.optimal_trajectory = new_trajectory

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
            self.drone.set_trajectory(self.optimal_trajectory)

    def optimal_trajectory(self):
        return self.optimal_trajectory
    

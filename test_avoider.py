import cv2
import numpy as np
from UAV_Object_Detection_And_Evasion_Layer import YOLOObstacleDetector, Avoider, Obstacle, DroneState, VideoFrameAnalyser, Runner

def generate_fake_obstacles():
    return [
        Obstacle(bbox=[100, 150, 180, 230], conf=0.9, class_name="object"),
        Obstacle(bbox=[300, 100, 360, 200], conf=0.85, class_name="object"),
        Obstacle(bbox=[250, 250, 330, 330], conf=0.80, class_name="object")
    ]

def test_blank():
    print("\n=== Running Avoider Test ===\n")

    # Create a blank frame for drawing
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Use fake obstacles for testing
    obstacles = generate_fake_obstacles()

    # Visualise the obstacles on the frame
    for obs in obstacles:
        cv2.rectangle(frame, 
                      (obs.bbox[0], obs.bbox[1]),
                      (obs.bbox[2], obs.bbox[3]),
                      (255, 255, 255), 2)

    # Starting drone state
    current_vector_3d = np.array([1.0, 0.2, -0.1])
    current_position = [0.0, 0.0, 0.0]

    drone = DroneState(current_position, current_vector_3d)

    # Screen + camera info
    screen_width = 200
    screen_height = 200
    fov = 100

    # Create avoider system
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

def test_obstacle1():
    print("\n=== Running Avoider Test 1 ===\n")
    runner = Runner()
    img_path = "obstacle1.jpg"
    current_vector_3d = np.array([209, 157, 100])
    runner.analyse_optimal_trajectory(img_path, current_vector_3d)

def test_obstacle2():
    print("\n=== Running Avoider Test 2 ===\n")
    runner = Runner()
    img_path = "obstacle2.jpg"
    current_vector_3d = np.array([1, 26, 202])
    runner.analyse_optimal_trajectory(img_path, current_vector_3d)

def test_obstacle3():
    print("\n=== Running Avoider Test 3 ===\n")
    runner = Runner()
    img_path = "obstacle3.jpg"
    current_vector_3d = np.array([1, 120, 67])
    runner.analyse_optimal_trajectory(img_path, current_vector_3d)

if __name__ == "__main__":
    test_obstacle1()
    test_obstacle2()
    test_obstacle3()

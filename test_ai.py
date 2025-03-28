import cv2
import numpy as np
import torch
import json
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# YOLOv5 wrapper
class YOLOv5:
    def __init__(self, weights, device):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True).to(device)
        self.device = device

    def predict(self, frame):
        results = self.model(frame)
        return results

# Load YOLOv5 model
model_path = "yolov5s.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLOv5(model_path, device=device)

# Input and output paths
video_path = "football_clip.mp4"
output_video_path = "output_football_clip.mp4"
results_file = "analysis_results.json"

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer for annotated output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Tracking variables
prev_positions = {}
player_ids = {}
ball_positions = []
next_id = 0
frame_count = 0
speed_data = []
possession_frames = 0
attempted_passes = 0
successful_passes = 0
BALL_CONF_THRESHOLD = 0.01
PLAYER_CONF_THRESHOLD = 0.4
prev_ball_owner = None
PIXELS_TO_METERS = 0.015
BALL_SIZE_THRESHOLD = 30
POSSESSION_THRESHOLD = 150
MAX_BALL_GAP = 15  
ball_frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = yolo_model.predict(frame)
    detections = results.pred[0]
    curr_positions = {}
    ball_candidates = []

    # Process detections
    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Player detection
        if int(cls) == 0 and conf > PLAYER_CONF_THRESHOLD and height > 40:
            if center_y < frame_height * 0.95:
                closest_id = None
                min_dist = float('inf')
                for pid, pos in prev_positions.items():
                    if pid.startswith("P_"):
                        dist = np.sqrt((center_x - pos[0])**2 + (center_y - pos[1])**2)
                        if dist < 100 and dist < min_dist:
                            min_dist = dist
                            closest_id = pid
                if closest_id is None:
                    closest_id = f"P_{next_id}"
                    next_id += 1
                label = f"Player_{closest_id[2:]}"
                curr_positions[closest_id] = [center_x, center_y]
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Ball detection for Any small object
        if conf > BALL_CONF_THRESHOLD and width < BALL_SIZE_THRESHOLD and height < BALL_SIZE_THRESHOLD:
            print(f"Frame {frame_count}: Potential ball - Conf: {conf:.2f}, Width: {width:.1f}, Height: {height:.1f}, Class: {int(cls)}")
            ball_candidates.append((center_x, center_y, conf, x_min, y_min, x_max, y_max))

    # Ball tracking with interpolation 
    ball_pos = None
    if ball_candidates:
        ball_frame_counter = 0
        ball_pos = max(ball_candidates, key=lambda x: x[2])  
        center_x, center_y, _, x_min, y_min, x_max, y_max = ball_pos
        obj_key = f"B_{int(center_x)}_{int(center_y)}"
        curr_positions[obj_key] = [center_x, center_y]
        ball_positions.append([center_x, center_y])
        if len(ball_positions) > 5:
            ball_positions.pop(0)
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        cv2.putText(frame, "Ball", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif ball_positions and ball_frame_counter < MAX_BALL_GAP:
        ball_frame_counter += 1
        last_ball = ball_positions[-1]
        obj_key = f"B_{int(last_ball[0])}_{int(last_ball[1])}"
        curr_positions[obj_key] = last_ball
        ball_pos = last_ball
        cv2.rectangle(frame, (int(last_ball[0] - 10), int(last_ball[1] - 10)), 
                      (int(last_ball[0] + 10), int(last_ball[1] + 10)), (0, 0, 255), 2)
        cv2.putText(frame, "Ball (interp)", (int(last_ball[0] - 10), int(last_ball[1] - 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Ball possession and passing logic
    curr_ball_owner = None
    if ball_pos:
        min_dist = float('inf')
        for obj_key, pos in curr_positions.items():
            if obj_key.startswith("P_"):
                dist = np.sqrt((pos[0] - ball_pos[0])**2 + (pos[1] - ball_pos[1])**2)
                if dist < POSSESSION_THRESHOLD:
                    curr_ball_owner = obj_key
                    possession_frames += 1
                    if prev_ball_owner and prev_ball_owner != curr_ball_owner:
                        attempted_passes += 1
                        successful_passes += 1
                    break
                elif prev_ball_owner and dist > POSSESSION_THRESHOLD * 2:
                    attempted_passes += 1
        prev_ball_owner = curr_ball_owner

    # Calculate speed for players
    for obj_key, curr_pos in curr_positions.items():
        if obj_key.startswith("P_") and obj_key in prev_positions:
            prev_pos = prev_positions[obj_key]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            speed = distance * fps * PIXELS_TO_METERS * 3.6
            speed_data.append(speed)

    prev_positions = curr_positions.copy()

    # Write annotated frame
    out.write(frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Calculate results
avg_player_speed = np.mean(speed_data) if speed_data else 0
passing_accuracy = (successful_passes / attempted_passes * 100) if attempted_passes > 0 else 0
ball_possession_time = possession_frames / fps if possession_frames > 0 else 0

# Save results to JSON >>> easier for me to handel in the front/back end
results = {
    "frame_count": frame_count,
    "player_speed_kmh": round(avg_player_speed, 2),
    "passing_accuracy_percent": round(passing_accuracy, 2),
    "ball_possession_time_sec": round(ball_possession_time, 2),
    "video_duration_sec": round(frame_count / fps, 4)
}
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4, cls=NumpyEncoder)

# Print results
print(f"Results saved to {results_file}:")
print(json.dumps(results, indent=4, cls=NumpyEncoder))
print(f"Annotated video saved to {output_video_path}")
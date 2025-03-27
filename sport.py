import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video file
cap = cv2.VideoCapture('/content/match_video.mp4')  # Provide the video path here

# Variables for passes and statistics
pass_detected = False
pass_count = 0
ball_touch_count = 0
speed = 0
previous_position = None

# Function to provide recommendations based on performance
def give_recommendations():
    recommendations = []
    
    # Passing analysis
    if pass_count < 3:
        recommendations.append("Try improving your passing technique.")
    
    # Speed analysis
    if speed < 3.0:
        recommendations.append("Increase your speed off the ball.")
    
    # Positioning analysis
    recommendations.append("Focus on better positioning on the field.")
    
    return recommendations

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame using MediaPipe
    results = pose.process(frame_rgb)

    # If body landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

        # Calculate player speed based on hand or foot movement
        if previous_position:
            current_position = (right_shoulder.x, right_shoulder.y)
            distance = math.sqrt((current_position[0] - previous_position[0])**2 + (current_position[1] - previous_position[1])**2)
            speed = distance * 30  # Estimate speed based on distance
        previous_position = (right_shoulder.x, right_shoulder.y)

        # Detect passes
        if right_wrist.x > right_shoulder.x and abs(right_wrist.y - right_shoulder.y) < 0.1:
            pass_detected = True
        if left_wrist.x < left_shoulder.x and abs(left_wrist.y - left_shoulder.y) < 0.1:
            pass_detected = True

        if pass_detected:
            pass_count += 1

        # Display pass count
        if pass_detected:
            cv2.putText(frame, f"Pass Count: {pass_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display ball touches count
        if right_wrist.x > left_wrist.x:  # Example condition for ball touch
            ball_touch_count += 1
            cv2.putText(frame, f"Ball Touches: {ball_touch_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display player speed
        cv2.putText(frame, f"Speed: {speed:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display recommendations on screen
        recommendations = give_recommendations()
        y_position = 200
        for recommendation in recommendations:
            cv2.putText(frame, recommendation, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_position += 40

    # Display the video
    cv2.imshow("Analyzing Video", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

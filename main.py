import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Load Video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Vehicle Classes
vehicle_classes = [
    'car',
    'motorcycle',
    'bus',
    'truck'
]

# Previous count
previous_vehicle_count = 0

# Store analytics data
traffic_data = []

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video Ended")
        break

    vehicle_count = 0

    emergency_detected = False
    wrong_direction_detected = False
    accident_detected = False

    # Frame dimensions
    height, width, _ = frame.shape

    center_x = width // 2

    # YOLO Detection
    results = model(frame)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            class_name = model.names[cls]

            if class_name in vehicle_classes:

                vehicle_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Vehicle label
                cv2.putText(
                    frame,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Emergency Detection
                if class_name == "bus":

                    emergency_detected = True

                    cv2.putText(
                        frame,
                        "EMERGENCY VEHICLE",
                        (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                # Wrong Direction Detection
                if x1 > center_x:

                    wrong_direction_detected = True

                    cv2.putText(
                        frame,
                        "WRONG DIRECTION",
                        (x1, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

    # Traffic Density
    if vehicle_count < 10:
        traffic_level = "LOW Traffic"

    elif vehicle_count < 25:
        traffic_level = "MEDIUM Traffic"

    else:
        traffic_level = "HIGH Traffic"

    # Accident Detection
    difference = abs(
        vehicle_count - previous_vehicle_count
    )

    if difference > 15:
        accident_detected = True

    previous_vehicle_count = vehicle_count

    # Signal Recommendation
    if emergency_detected:

        signal_recommendation = (
            "PRIORITIZE EMERGENCY LANE"
        )

    elif traffic_level == "HIGH Traffic":

        signal_recommendation = (
            "Increase GREEN Signal Time"
        )

    elif traffic_level == "MEDIUM Traffic":

        signal_recommendation = (
            "Maintain Normal Signal Timing"
        )

    else:

        signal_recommendation = (
            "Reduce GREEN Signal Time"
        )

    # Save Analytics Data
    traffic_data.append({

        "Time":
        datetime.now().strftime("%H:%M:%S"),

        "Vehicle_Count":
        vehicle_count,

        "Traffic_Level":
        traffic_level,

        "Signal_Recommendation":
        signal_recommendation,

        "Emergency_Detected":
        emergency_detected,

        "Wrong_Direction":
        wrong_direction_detected,

        "Accident_Detected":
        accident_detected
    })

    # Display Count
    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Traffic Level
    cv2.putText(
        frame,
        f"Traffic Level: {traffic_level}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    # Signal Recommendation
    cv2.putText(
        frame,
        f"Signal: {signal_recommendation}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Emergency Alert
    if emergency_detected:

        cv2.putText(
            frame,
            "EMERGENCY ALERT ACTIVATED",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    # Wrong Direction Alert
    if wrong_direction_detected:

        cv2.putText(
            frame,
            "WRONG-WAY VEHICLE DETECTED",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    # Accident Alert
    if accident_detected:

        cv2.putText(
            frame,
            "POSSIBLE ACCIDENT DETECTED",
            (20, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    # Show Frame
    cv2.imshow(
        "Autonomous Traffic Intelligence Platform",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert to DataFrame
df = pd.DataFrame(traffic_data)

# Save CSV
df.to_csv(
    "data/traffic_data.csv",
    index=False
)

print("CSV Analytics Saved Successfully!")

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load Trained 15-Class Model
model = tf.keras.models.load_model("shape_model.h5")

DIGIT_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
SHAPE_CLASSES = ['circle', 'triangle', 'square', 'star', 'line']
ALL_CLASSES = DIGIT_CLASSES + SHAPE_CLASSES

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# EMA Trajectory Smoothing Filter
alpha = 0.35
def apply_ema_filter(current_pt, prev_pt, alpha_val=0.35):
    if prev_pt == (0, 0):
        return current_pt
    return (int(alpha_val * current_pt[0] + (1 - alpha_val) * prev_pt[0]),
            int(alpha_val * current_pt[1] + (1 - alpha_val) * prev_pt[1]))

# Predict drawn input via CNN
def process_and_classify_stroke(raw_canvas):
    coords = cv2.findNonZero(raw_canvas)
    if coords is None:
        return None, None
    
    x, y, w, h = cv2.boundingRect(coords)
    if w < 15 or h < 15:
        return None, None

    # Crop stroke and pad into 1:1 square aspect ratio
    roi = raw_canvas[y:y+h, x:x+w]
    max_dim = max(w, h)
    square_padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
    pad_x, pad_y = (max_dim - w) // 2, (max_dim - h) // 2
    square_padded[pad_y:pad_y+h, pad_x:pad_x+w] = roi

    # Resize to 64x64 and normalize
    resized_input = cv2.resize(square_padded, (64, 64))
    input_tensor = resized_input.astype('float32') / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=(0, -1))

    predictions = model.predict(input_tensor, verbose=0)
    class_idx = np.argmax(predictions[0])
    predicted_label = ALL_CLASSES[class_idx]
    
    return predicted_label, (x, y, w, h)

# Rendering Engine: Handles Digits vs. Shapes
def render_prediction_result(canvas, label, bbox):
    x, y, w, h = bbox
    center = (x + w // 2, y + h // 2)
    roi = canvas[y:y+h, x:x+w].copy()
    
    # Clear messy drawing stroke from canvas
    canvas[y:y+h, x:x+w] = 0

    # Handler A: Digit Output (Render Clean Digital Text)
    if label in DIGIT_CLASSES:
        font_scale = max(w, h) / 35.0
        cv2.putText(canvas, label, (x + 5, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, 3)

    # Handler B: Geometric Shape Output (Vector Snapping)
    else:
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        main_contour = max(contours, key=cv2.contourArea) + np.array([x, y])

        if label == 'circle':
            (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
            cv2.circle(canvas, (int(cx), int(cy)), int(radius), 255, 3)

        elif label == 'line':
            [vx, vy, x0, y0] = cv2.fitLine(main_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            pt1 = (int(x0 - vx * 100), int(y0 - vy * 100))
            pt2 = (int(x0 + vx * 100), int(y0 + vy * 100))
            cv2.line(canvas, pt1, pt2, 255, 3)

        else:
            # Polygons: Triangle, Square, Star
            perimeter = cv2.arcLength(main_contour, True)
            epsilon = 0.03 * perimeter
            approx_polygon = cv2.approxPolyDP(main_contour, epsilon, True)
            cv2.polylines(canvas, [approx_polygon], isClosed=True, color=255, thickness=3)

# Main Loop
cap = cv2.VideoCapture(0)
canvas = None
prev_point, current_smooth = (0, 0), (0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            idx_tip = hand_landmarks.landmark[8]
            mid_tip = hand_landmarks.landmark[12]

            raw_x, raw_y = int(idx_tip.x * frame_w), int(idx_tip.y * frame_h)
            mid_y = int(mid_tip.y * frame_h)

            current_smooth = apply_ema_filter((raw_x, raw_y), current_smooth, alpha)

            idx_pip_y = int(hand_landmarks.landmark[6].y * frame_h)
            mid_pip_y = int(hand_landmarks.landmark[10].y * frame_h)

            # Draw Mode: Index Finger Up, Middle Finger Down
            if raw_y < idx_pip_y and mid_y > mid_pip_y:
                cv2.circle(frame, current_smooth, 5, (0, 255, 0), -1)
                if prev_point != (0, 0):
                    cv2.line(canvas, prev_point, current_smooth, 255, 4)
                prev_point = current_smooth
            else:
                prev_point = (0, 0)

    # Overlay Canvas
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    canvas_bgr[:, :, 1] = canvas  # Green strokes
    combined_display = cv2.addWeighted(frame, 0.7, canvas_bgr, 0.8, 0)

    cv2.imshow("Air-GR: Digits & Shapes Predictor", combined_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((frame_h, frame_w), dtype=np.uint8)
        prev_point = (0, 0)
    elif key == ord('p'):
        label, bbox = process_and_classify_stroke(canvas)
        if label and bbox:
            render_prediction_result(canvas, label, bbox)

cap.release()
cv2.destroyAllWindows()
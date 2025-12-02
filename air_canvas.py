import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Color choices (BGR format)
colors = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'black': (0, 0, 0),
    'eraser': (255, 255, 255)
}
color_names = list(colors.keys())
current_color = colors['black']

# Initialize variables
prev_x, prev_y = None, None
canvas = None
pen_thickness = 5
drawing = False
undo_stack = []

# Dimensions for the color selection rectangles and "Erase All" button
color_rect_width = 60
color_rect_height = 60
erase_all_button_width = 120
erase_all_button_height = 60

# Pen/Eraser toggle indicator
is_pen_active = True

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(rgb_frame)

    # Initialize the canvas if it is None
    if canvas is None:
        canvas = np.ones_like(frame) * 255  # White canvas

    # Draw color selection boxes at the top of the frame
    for i, color_name in enumerate(color_names):
        color = colors[color_name]
        cv2.rectangle(frame, (i * color_rect_width, 0),
                      ((i + 1) * color_rect_width, color_rect_height), color, -1)
        cv2.putText(frame, color_name, (i * color_rect_width + 10, color_rect_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if color_name != 'eraser' else (0, 0, 255), 2)

    # Draw "Erase All" button
    cv2.rectangle(frame, (w - erase_all_button_width, 0),
                  (w, erase_all_button_height), (0, 0, 0), -1)
    cv2.putText(frame, "Erase All", (w - erase_all_button_width + 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display pen thickness, current tool, and tool symbol
    thickness_display_x = w - 200
    thickness_display_y = h - 50
    cv2.putText(frame, f"Thickness: {pen_thickness}", (thickness_display_x, thickness_display_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    tool_text = "Tool: Pen" if is_pen_active else "Tool: Eraser"
    cv2.putText(frame, tool_text, (thickness_display_x, thickness_display_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Add symbols based on the tool selected
    if is_pen_active:
        cv2.circle(frame, (thickness_display_x + 150, thickness_display_y + 15), 10, current_color, -1)
    else:
        cv2.putText(frame, "Eraser", (thickness_display_x + 130, thickness_display_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the tip of the index finger, baby finger, and thumb
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            little_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            bx, by = int(little_finger_tip.x * w), int(little_finger_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Toggle drawing mode when index finger and thumb click together
            thumb_index_distance = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)
            if thumb_index_distance < 30:  # Adjusted threshold
                drawing = not drawing
                prev_x, prev_y = None, None  # Reset previous coordinates when toggling

            # Select color or erase all with baby finger
            if by < color_rect_height:
                selected_index = bx // color_rect_width
                if selected_index < len(color_names):
                    current_color = colors[color_names[selected_index]]
                    is_pen_active = current_color != colors['eraser']
                elif w - erase_all_button_width <= bx < w and by < erase_all_button_height:
                    undo_stack.clear()  # Clear undo stack when erasing all
                    canvas = np.ones_like(frame) * 255  # Erase all

            # Adjust pen thickness
            if is_pen_active and by < color_rect_height:
                pen_thickness = 7 if current_color != colors['eraser'] else 20  # Adjust thickness

            # Draw on the canvas with the index finger if drawing mode is on
            if drawing and prev_x is not None and prev_y is not None:
                # Stabilize drawing with a small movement threshold
                if np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2) > 5:  # Adjust threshold as needed
                    undo_stack.append(canvas.copy())  # Save the current state for undo
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), current_color, pen_thickness)

            prev_x, prev_y = cx, cy

            # Draw landmarks on the frame (for visualization)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None

    # Combine the canvas and the video frame
    combined_frame = np.hstack((frame, canvas))

    # Display the frame
    cv2.imshow("Air Canvas with Whiteboard", combined_frame)

    # Undo the last action if 'u' is pressed
    if cv2.waitKey(1) & 0xFF == ord('u') and undo_stack:
        canvas = undo_stack.pop()

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

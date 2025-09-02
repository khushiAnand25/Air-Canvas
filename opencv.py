import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# ---------------------------- Initialize Colors and Tools ----------------------------
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
points = [bpoints, gpoints, rpoints, ypoints]

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)]  # Last color = Clear
labels = ["BLUE", "GREEN", "RED", "YELLOW", "CLEAR"]
colorIndex = 0
current_tool = "Select Tool"

# ---------------------------- Load tools image (handle PNG transparency) ----------------------------
tools = cv2.imread(r"Import your tool.png file here", cv2.IMREAD_UNCHANGED)
if tools.shape[2] == 4:  # If has alpha channel
    bgr = tools[:, :, :3]
    alpha = tools[:, :, 3] / 255.0
    white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
    tools = (bgr * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)

# ---------------------------- Initialize Mediapipe ----------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# ---------------------------- Setup Camera ----------------------------
cap = cv2.VideoCapture(0)

# ---------------------------- Tool Selection Variables ----------------------------
ml = 150
max_x, max_y = 250 + ml, 50
var_inits = False
thick = 4
prevx, prevy = 0, 0
prev_point = None
last_tool_change_time = time.time()
tool_change_cooldown = 0.5  # 500 ms

# ---------------------------- Helper Functions ----------------------------
def index_raised(yi, y9):
    return (y9 - yi) > 40

def get_tool(x):
    if x < 50 + ml:
        return "Line"
    elif x < 100 + ml:
        return "Rectangle"
    elif x < 150 + ml:
        return "Draw"
    elif x < 200 + ml:
        return "Circle"
    else:
        return "Erase"

# ---------------------------- Main Loop ----------------------------
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Initialize canvas
    if 'paintWindow' not in globals() or paintWindow.shape[:2] != frame.shape[:2]:
        paintWindow = np.ones_like(frame) * 255

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # ---------------------------- Draw Color Palette ----------------------------
    color_positions = [40, 140, 240, 340, 440]
    for i, y_pos in enumerate(color_positions):
        color = colors[i]
        cv2.rectangle(frame, (1, y_pos), (65, y_pos + 65), color, -1)
        cv2.putText(frame, labels[i], (10, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    # Highlight selected color
    cv2.rectangle(frame, (1, color_positions[colorIndex]), (65, color_positions[colorIndex] + 65), (0,0,0), 3)

    # ---------------------------- Place Tool Selection ----------------------------
    tools_resized = cv2.resize(tools, (max_x - ml, max_y))
    frame[:max_y, ml:max_x] = cv2.addWeighted(tools_resized, 0.7, frame[:max_y, ml:max_x], 0.3, 0)

    # ---------------------------- Process Hand Landmarks ----------------------------
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in handLms.landmark]
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            fore_finger = landmarks[8]  # Index finger tip
            cv2.circle(frame, fore_finger, 5, (0, 255, 0), -1)

            # ---------------------------- Tool Selection ----------------------------
            if fore_finger[1] < max_y:
                current_time = time.time()
                if current_time - last_tool_change_time > tool_change_cooldown:
                    last_tool_change_time = current_time
                    current_tool = get_tool(fore_finger[0])
                    print("Selected Tool:", current_tool)
                    prevx, prevy = 0, 0
                    prev_point = None

            # ---------------------------- Color Selection ----------------------------
            elif fore_finger[0] < 65:
                y = fore_finger[1]
                if 40 <= y <= 105:
                    colorIndex = 0
                elif 140 <= y <= 205:
                    colorIndex = 1
                elif 240 <= y <= 305:
                    colorIndex = 2
                elif 340 <= y <= 405:
                    colorIndex = 3
                elif 440 <= y <= 505:
                    paintWindow[:] = 255
                    print("Canvas Cleared")

            # ---------------------------- Drawing Tools ----------------------------
            xi, yi = fore_finger
            y9 = landmarks[9][1]

            if current_tool == "Draw":
                if prevx == 0 and prevy == 0:
                    prevx, prevy = fore_finger
                elif index_raised(yi, y9):
                    cv2.line(paintWindow, (prevx, prevy), fore_finger, colors[colorIndex], thick)
                    prevx, prevy = fore_finger
                else:
                    prevx, prevy = fore_finger

            elif current_tool == "Line":
                if prev_point is None:
                    prev_point = fore_finger
                elif not index_raised(yi, y9):
                    cv2.line(paintWindow, prev_point, fore_finger, colors[colorIndex], thick)
                    prev_point = None

            elif current_tool == "Rectangle":
                if prev_point is None:
                    prev_point = fore_finger
                else:
                    # Live preview
                    paintWindow_copy = paintWindow.copy()
                    cv2.rectangle(paintWindow_copy, prev_point, fore_finger, colors[colorIndex], thick)
                    combined_frame = cv2.addWeighted(frame, 0.5, paintWindow_copy, 0.5, 0)
                    cv2.imshow("Hand Gesture Drawing", combined_frame)

                    if not index_raised(yi, y9):
                        cv2.rectangle(paintWindow, prev_point, fore_finger, colors[colorIndex], thick)
                        prev_point = None

            elif current_tool == "Circle":
                if prev_point is None:
                    prev_point = fore_finger
                else:
                    radius = int(((prev_point[0]-fore_finger[0])**2 + (prev_point[1]-fore_finger[1])**2)**0.5)
                    paintWindow_copy = paintWindow.copy()
                    cv2.circle(paintWindow_copy, prev_point, radius, colors[colorIndex], thick)
                    combined_frame = cv2.addWeighted(frame, 0.5, paintWindow_copy, 0.5, 0)
                    cv2.imshow("Hand Gesture Drawing", combined_frame)

                    if not index_raised(yi, y9):
                        cv2.circle(paintWindow, prev_point, radius, colors[colorIndex], thick)
                        prev_point = None

            elif current_tool == "Erase":
                eraser_size = 20
                cv2.circle(paintWindow, fore_finger, eraser_size, (255, 255, 255), -1)

    # ---------------------------- Overlay Canvas ----------------------------
    combined_frame = cv2.addWeighted(frame, 0.5, paintWindow, 0.5, 0)
    cv2.imshow("Hand Gesture Drawing", combined_frame)

    # ---------------------------- Exit ----------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

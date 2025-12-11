from handTracker import *
import cv2
import mediapipe as mp
import numpy as np
import random
import time
last_save_time = 0  # Track last save time
class ColorRect():
    def __init__(self, x, y, w, h, color, text='', icon_path=None, alpha=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha
        self.icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED) if icon_path else None  # Load icon if provided

    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        # Draw the box with transparency
        alpha = self.alpha
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]
        overlay = np.ones(bg_rec.shape, dtype=np.uint8) * np.array(self.color, dtype=np.uint8)
        res = cv2.addWeighted(bg_rec, alpha, overlay, 1 - alpha, 1.0)
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res

        # Draw icon if available
        if self.icon is not None:
            icon_resized = cv2.resize(self.icon, (self.w - 20, self.h - 20))  # Resize icon
            x_offset = self.x + 10
            y_offset = self.y + 10

            # Handle transparent PNGs
            if self.icon.shape[2] == 4:  # If image has alpha channel
                for c in range(3):  # Loop through color channels (B, G, R)
                    img[y_offset:y_offset + icon_resized.shape[0], x_offset:x_offset + icon_resized.shape[1], c] = (
                        icon_resized[:, :, c] * (icon_resized[:, :, 3] / 255.0) +
                        img[y_offset:y_offset + icon_resized.shape[0], x_offset:x_offset + icon_resized.shape[1], c] * (1 - icon_resized[:, :, 3] / 255.0)
                    )
            else:  # No alpha channel
                img[y_offset:y_offset + icon_resized.shape[0], x_offset:x_offset + icon_resized.shape[1]] = icon_resized
        else:
            # Draw text if no icon is present
            text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
            text_pos = (int(self.x + self.w / 2 - text_size[0][0] / 2), int(self.y + self.h / 2 + text_size[0][1] / 2))
            cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        return self.x < x < self.x + self.w and self.y < y < self.y + self.h


#initilize the habe detector
detector = HandTracker(detectionCon=0.8)

#initilize the camera 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# creating canvas to draw on it
canvas = np.zeros((720,1280,3), np.uint8)

# define a previous point to be used with drawing a line
px,py = 0,0
#initial brush color
color = (255,0,0)
#####
brushSize = 5
eraserSize = 20
####

########### creating colors ########
# Colors button
colorsBtn = ColorRect(130, 10, 60, 60, (120,255,0), icon_path="icons/paint-bucket.png")




colors = []
#random color
b = int(random.random()*255)-1
g = int(random.random()*255)
r = int(random.random()*255)
print(b,g,r)
colors.append(ColorRect(190,10,60,60, (b,g,r)))
#red
colors.append(ColorRect(250,10,60,60, (0,0,255)))
#blue
colors.append(ColorRect(310,10,60,60, (255,0,0)))
#green
colors.append(ColorRect(370,10,60,60, (0,255,0)))
#yellow
colors.append(ColorRect(430,10,60,60, (0,255,255)))
#erase (black)
colors.append(ColorRect(490,10,60,60, (0,0,0), icon_path="icons/rubber.png"))

#clear
clear = ColorRect(550,10,60,60, (100,100,100), icon_path="icons/bin.png")

########## pen sizes #######
pens = []
for i, penSize in enumerate(range(5,25,5)):
    pens.append(ColorRect(70,70+60*i,60,60, (50,50,50), str(penSize)))

penBtn = ColorRect(70, 10, 60, 60, (255,255,255), icon_path="icons/pencil.png")

# white board button
boardBtn = ColorRect(10, 10, 60, 60, (255,255,255), icon_path="icons/easel.png")

#define a white board to draw on
whiteBoard = ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 0.6)

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True

#Drawing Part
#
#
#
# Global variables to track the last time each gesture was detected
last_heart_time = 0
last_circle_time = 0
last_ok_time = 0

# Cooldown period in seconds
COOLDOWN = 2  # Adjust as needed
#Drawing circle

def draw_circle(canvas, center_x, center_y, radius=50, color=(0, 255, 0), thickness=2):
    cv2.circle(canvas, (center_x, center_y), radius, color, thickness)

#drawing heart
def draw_heart(canvas, center_x, center_y, size=50, color=(255, 0, 0), thickness=2):
    # Define the points for the heart shape
    pts = np.array([
        [center_x, center_y],
        [center_x - size, center_y - size],
        [center_x - size*2, center_y],
        [center_x, center_y + size*2],
        [center_x + size*2, center_y],
        [center_x + size, center_y - size]
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))  # Reshaping to make it compatible with polylines

    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

#heart gesture
#
#
import numpy as np

def is_heart_gesture(positions):
    if len(positions) < 21:  # Ensure all 21 landmarks for one hand are detected
        return False
    
    # Landmarks for the single hand
    thumb_tip = positions[4]   # Thumb tip
    index_tip = positions[8]   # Index finger tip
    middle_tip = positions[12] # Middle finger tip
    ring_tip = positions[16]   # Ring finger tip
    pinky_tip = positions[20]  # Pinky tip

    # Calculate distances between key points
    thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
    thumb_middle_dist = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))
    index_middle_dist = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))
    middle_ring_dist = np.linalg.norm(np.array(middle_tip) - np.array(ring_tip))
    ring_pinky_dist = np.linalg.norm(np.array(ring_tip) - np.array(pinky_tip))

    # Conditions for different heart gestures
    # 🫰: Thumb and index finger tips are close, forming a small heart
    gesture_1 = thumb_index_dist < 50 and index_middle_dist > 50

    # 🫶: Thumb is close to both index and middle fingers
    #gesture_2 = thumb_index_dist < 50 and thumb_middle_dist < 50

    # 🤟: Thumb and index finger form a circle, other fingers are extended
    gesture_3 = thumb_index_dist < 50 and middle_ring_dist > 50 and ring_pinky_dist > 50

    # Return True if any of the heart gestures are detected
    return gesture_1 or gesture_3
#two hands heart
def is_two_handed_heart_gesture(left_positions, right_positions):
    """
    Detects the two-handed heart gesture (🫶).

    Parameters:
        left_positions (list): Landmarks for the left hand.
        right_positions (list): Landmarks for the right hand.

    Returns:
        bool: True if the gesture is detected, False otherwise.
    """
    if len(left_positions) < 21 or len(right_positions) < 21:  # Ensure landmarks for both hands are detected
        return False
    
    # Left hand landmarks
    left_thumb_tip = left_positions[4]   # Thumb tip
    left_index_tip = left_positions[8]   # Index finger tip
    left_middle_tip = left_positions[12] # Middle finger tip
    left_ring_tip = left_positions[16]   # Ring finger tip
    left_pinky_tip = left_positions[20]  # Pinky tip

    # Right hand landmarks
    right_thumb_tip = right_positions[4]   # Thumb tip
    right_index_tip = right_positions[8]   # Index finger tip
    right_middle_tip = right_positions[12] # Middle finger tip
    right_ring_tip = right_positions[16]   # Ring finger tip
    right_pinky_tip = right_positions[20]  # Pinky tip

    # Check if thumbs and index fingers of both hands are close together
    left_heart_condition = np.linalg.norm(np.array(left_thumb_tip) - np.array(left_index_tip)) < 30
    right_heart_condition = np.linalg.norm(np.array(right_thumb_tip) - np.array(right_index_tip)) < 30

    # Check if the hands are close to each other (forming a heart)
    hands_close = np.linalg.norm(np.array(left_index_tip) - np.array(right_index_tip)) < 50

    # Check if middle, ring, and pinky fingers are extended (not touching)
    left_fingers_extended = (
        np.linalg.norm(np.array(left_middle_tip) - np.array(left_ring_tip)) > 30 and
        np.linalg.norm(np.array(left_ring_tip) - np.array(left_pinky_tip)) > 30
    )
    right_fingers_extended = (
        np.linalg.norm(np.array(right_middle_tip) - np.array(right_ring_tip)) > 30 and
        np.linalg.norm(np.array(right_ring_tip) - np.array(right_pinky_tip)) > 30
    )

    return (
        left_heart_condition and
        right_heart_condition and
        hands_close and
        left_fingers_extended and
        right_fingers_extended
    )

#o gesture detecture
def is_o_gesture(positions):
    if len(positions) < 21:  # Ensure all landmarks are detected
        return False
    
    index_tip = positions[8]   # Index finger tip
    middle_tip = positions[12] # Middle finger tip
    ring_tip = positions[16]   # Ring finger tip
    pinky_tip = positions[20]  # Pinky tip
    thumb_tip = positions[4]   # Thumb tip

    # Check if index, middle, ring, and pinky fingers are close together
    fingers_touching = (
        np.linalg.norm(np.array(index_tip) - np.array(middle_tip)) < 30 and
        np.linalg.norm(np.array(middle_tip) - np.array(ring_tip)) < 30 and
        np.linalg.norm(np.array(ring_tip) - np.array(pinky_tip)) < 30
    )

    # Check if thumb is touching the tips of the four fingers
    thumb_touching = (
        np.linalg.norm(np.array(thumb_tip) - np.array(index_tip)) < 50 or
        np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip)) < 50 or
        np.linalg.norm(np.array(thumb_tip) - np.array(ring_tip)) < 50 or
        np.linalg.norm(np.array(thumb_tip) - np.array(pinky_tip)) < 50
    )

    return fingers_touching and thumb_touching


#ok gesture
def is_ok_gesture(positions):
    if len(positions) < 21:
        return False  # Ensure all 21 landmarks are detected

    thumb_tip = positions[4]
    index_tip = positions[8]
    middle_tip = positions[12]
    ring_tip = positions[16]
    pinky_tip = positions[20]

    # Check if thumb tip and index tip are close together
    distance_thumb_index = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
    touching_threshold = 30  # Adjust based on your camera resolution

    # Check if other fingers are extended
    if (distance_thumb_index < touching_threshold and
        middle_tip[1] < positions[9][1] and
        ring_tip[1] < positions[13][1] and
        pinky_tip[1] < positions[17][1]):
        return True
    
    return False



while True:
    if coolingCounter:
        coolingCounter -= 1

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    # Detect hands
    detector.findHands(frame)
    upFingers = detector.getUpFingers(frame)
    hands = detector.results.multi_hand_landmarks
    current_time = time.time()

    # Initialize left and right hand positions
    left_positions = None
    right_positions = None
    positions = None  # For single-hand gestures

    if hands:
        for i, hand_landmarks in enumerate(hands):
            # Check if the hand is left or right
            handedness = detector.results.multi_handedness[i].classification[0].label
            positions = detector.getPosition(frame, handNo=i, draw=False)

            if handedness == "Left":
                left_positions = positions
            elif handedness == "Right":
                right_positions = positions

    # Check for two-handed heart gesture
    if left_positions and right_positions:  # Ensure both hands are detected
        if is_two_handed_heart_gesture(left_positions, right_positions) and coolingCounter == 0:
            print("Two-Handed Heart Gesture Detected! Drawing Heart...")

            # Calculate the center between the two index finger tips
            left_index_tip = left_positions[8]  # Index finger tip of the left hand
            right_index_tip = right_positions[8]  # Index finger tip of the right hand

            center_x = (left_index_tip[0] + right_index_tip[0]) // 2  # Midpoint X
            center_y = (left_index_tip[1] + right_index_tip[1]) // 2  # Midpoint Y

            # Draw the heart at the calculated center
            draw_heart(canvas, center_x, center_y, size=50)

            # Reset the cooldown counter
            coolingCounter = 30  # Adjust as needed

    # Check for one-hand gestures (O gesture and OK gesture)
    if positions:  # Ensure single-hand landmarks are detected
        # O gesture detection

        if is_o_gesture(positions) and (current_time - last_circle_time) > COOLDOWN:
            print("O Gesture Detected! Drawing Circle...")
            center_x, center_y = positions[8]  # Place circle at index finger tip
            draw_circle(canvas, center_x, center_y, radius=50, color=(0, 255, 0), thickness=2)
            last_circle_time = current_time  # Update the last detection time

        elif is_heart_gesture(positions) and (current_time - last_circle_time) > COOLDOWN:
            print("heart Gesture Detected! Drawing Circle...")
            center_x, center_y = positions[8]  # Place circle at index finger tip
            draw_heart(canvas, center_x, center_y, size=50)
            last_circle_time = current_time  # Update the last detection time

        # OK gesture detection
        elif is_ok_gesture(positions) and (current_time - last_ok_time) > COOLDOWN:
            print("OK Gesture Detected! Saving Drawing...")
            cv2.imwrite('saved_drawing.jpg', canvas)
            last_ok_time = current_time  # Update the last detection time

    # Drawing logic
    if upFingers and positions:  # Ensure positions are available
        x, y = positions[8][0], positions[8][1]

        if upFingers[1] and not whiteBoard.isOver(x, y):
            px, py = 0, 0

            ##### pen sizes ######
            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x, y):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            ####### chose a color for drawing #######
            if not hideColors:
                for cb in colors:
                    if cb.isOver(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                # Clear 
                if clear.isOver(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720, 1280, 3), np.uint8)
                else:
                    clear.alpha = 0.5

            # Color button
            if colorsBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                colorsBtn.alpha = 0
                hideColors = False if hideColors else True
                colorsBtn.text = 'Colors' if hideColors else 'Hide'
            else:
                colorsBtn.alpha = 0.5

            # Pen size button
            if penBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                penBtn.alpha = 0
                hidePenSizes = False if hidePenSizes else True
                penBtn.text = 'Pen' if hidePenSizes else 'Hide'
            else:
                penBtn.alpha = 0.5

            # Whiteboard button
            if boardBtn.isOver(x, y) and not coolingCounter:
                coolingCounter = 10
                boardBtn.alpha = 0
                hideBoard = False if hideBoard else True
                boardBtn.text = 'Board' if hideBoard else 'Hide'
            else:
                boardBtn.alpha = 0.5

        elif upFingers[1] and not upFingers[2]:
            if whiteBoard.isOver(x, y) and not hideBoard:
                cv2.circle(frame, positions[8], brushSize, color, -1)
                # Drawing on the canvas
                if px == 0 and py == 0:
                    px, py = positions[8]
                if color == (0, 0, 0):
                    cv2.line(canvas, (px, py), positions[8], color, eraserSize)
                else:
                    cv2.line(canvas, (px, py), positions[8], color, brushSize)
                px, py = positions[8]
        else:
            px, py = 0, 0

    # Put colors button
    colorsBtn.drawRect(frame)
    cv2.rectangle(frame, (colorsBtn.x, colorsBtn.y), (colorsBtn.x + colorsBtn.w, colorsBtn.y + colorsBtn.h), (255, 255, 255), 2)

    # Put whiteboard button
    boardBtn.drawRect(frame)
    cv2.rectangle(frame, (boardBtn.x, boardBtn.y), (boardBtn.x + boardBtn.w, boardBtn.y + boardBtn.h), (255, 255, 255), 2)

    # Put the whiteboard on the frame
    if not hideBoard:
        whiteBoard.drawRect(frame)
        ########### Moving the draw to the main image #########
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

    ########## Pen colors' boxes #########
    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)

        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

    ########## Brush size boxes ######
    penBtn.color = color
    penBtn.drawRect(frame)
    cv2.rectangle(frame, (penBtn.x, penBtn.y), (penBtn.x + penBtn.w, penBtn.y + penBtn.h), (255, 255, 255), 2)
    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

    cv2.imshow('video', frame)
    k = cv2.waitKey(1)
    if k == ord('c'):  # Clear screen (show black)
        blank = np.zeros_like(frame)
        cv2.imshow('video', blank)
        print("Screen cleared!")
    elif k == ord('q'):  # Quit program
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
import random
import mediapipe as mp

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOJI_DIR = os.path.join(BASE_DIR, "emojis")

def load_emoji(filename, size=(120, 120)):
    path = os.path.join(EMOJI_DIR, filename)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Emoji not found at {path}")

    return cv2.resize(img, size)


EMOJI_IMG = {
    "Rock": load_emoji("rock.jpg"),
    "Paper": load_emoji("paper.png"),
    "Scissors": load_emoji("scissors.png"),
    None: None
}


def overlay_png(bg, fg, x, y):
    if fg is None:
        return bg

    h, w = fg.shape[:2]

    # If image has alpha channel
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (
                alpha * fg[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )
    else:
        # No alpha channel → simple overlay
        bg[y:y+h, x:x+w] = fg

    return bg


# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Game State 
MOVES = ["Rock", "Paper", "Scissors"]
ai_move = None
human_move = None
winner = None

ai_score = 0
human_score = 0

start_time = time.time()
moves_locked = False


# Helper Functions 
def get_finger_states(hand_landmarks):
    fingers = {}

    # Thumb (ignore for gesture)
    fingers["thumb"] = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x

    # Other fingers
    fingers["index"] = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    fingers["middle"] = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    fingers["ring"] = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    fingers["pinky"] = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

    return fingers


def finger_to_move(fingers):
    if not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "Rock"

    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "Scissors"

    if fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
        return "Paper"

    return None



def decide_winner(player, ai):
    if player == ai:
        return "Draw"
    if (
        (player == "Rock" and ai == "Scissors") or
        (player == "Paper" and ai == "Rock") or
        (player == "Scissors" and ai == "Paper")
    ):
        return "You"
    return "AI"


# Main Loop 
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    detected_move = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_states(hand_landmarks)
            detected_move = finger_to_move(fingers)

    elapsed = time.time() - start_time

    # Countdown
    if elapsed < 3:
        countdown = 3 - int(elapsed)
        cv2.putText(
            frame, str(countdown),
            (w // 2 - 30, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            4, (0, 0, 255), 6
        )

    # Lock Moves 
    elif not moves_locked:
        ai_move = random.choice(MOVES)
        human_move = detected_move
        winner = decide_winner(human_move, ai_move)

        if winner == "You":
            human_score += 1
        elif winner == "AI":
            ai_score += 1

        moves_locked = True

    # Display Result 
    else:
        cv2.putText(frame,f"AI: {ai_move}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)
        frame = overlay_png(frame, EMOJI_IMG[ai_move], 50, 120)

        cv2.putText(frame, f"You: {human_move}", (w - 350, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        frame = overlay_png(frame, EMOJI_IMG[human_move], w - 200, 120)

        cv2.putText(frame, f"Winner: {winner}",
                    (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

        cv2.putText(frame, f"Score  You: {human_score}  AI: {ai_score}",
                    (w // 2 - 250, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Auto reset
        if elapsed > 8:
            start_time = time.time()
            moves_locked = False
            ai_move = None
            human_move = None
            winner = None

    cv2.imshow("Rock Paper Scissors - AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

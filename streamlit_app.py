import secrets
import threading
import time
from pathlib import Path

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


BASE_DIR = Path(__file__).resolve().parent
MOVES = ("Rock", "Paper", "Scissors")
ROUND_SECONDS = 3
RESULT_SECONDS = 4


def decide_winner(player, opponent):
    if player == opponent:
        return "Draw"
    if (player, opponent) in (
        ("Rock", "Scissors"),
        ("Paper", "Rock"),
        ("Scissors", "Paper"),
    ):
        return "You win"
    return "AI wins"


class RockPaperScissorsProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._draw = mp.solutions.drawing_utils
        self._hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        self._emoji_images = {
            move: self._load_emoji(filename)
            for move, filename in {
                "Rock": "rock.jpg",
                "Paper": "paper.png",
                "Scissors": "scissors.png",
            }.items()
        }
        self._phase = "ready"
        self._round_started_at = 0.0
        self._result_started_at = 0.0
        self._human_move = None
        self._ai_move = None
        self._winner = None
        self._message = "Press Play round to begin"
        self._human_score = 0
        self._ai_score = 0

    @staticmethod
    def _load_emoji(filename):
        image = cv2.imread(str(BASE_DIR / "emojis" / filename), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not load emojis/{filename}")
        return cv2.resize(image, (88, 88))

    def start_round(self):
        with self._lock:
            if self._phase != "ready":
                return False
            self._phase = "countdown"
            self._round_started_at = time.monotonic()
            self._message = "Show Rock, Paper, or Scissors"
            return True

    def snapshot(self):
        with self._lock:
            return {
                "phase": self._phase,
                "message": self._message,
                "human_score": self._human_score,
                "ai_score": self._ai_score,
            }

    @staticmethod
    def _get_move(hand_landmarks):
        landmarks = hand_landmarks.landmark
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y

        if not any((index_up, middle_up, ring_up, pinky_up)):
            return "Rock"
        if index_up and middle_up and not ring_up and not pinky_up:
            return "Scissors"
        if index_up and middle_up and ring_up and pinky_up:
            return "Paper"
        return None

    @staticmethod
    def _overlay_image(frame, image, x, y):
        height, width = image.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        x = max(0, min(x, frame_width - width))
        y = max(0, min(y, frame_height - height))
        region = frame[y : y + height, x : x + width]
        if image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype("float32") / 255.0
            region[:] = (alpha * image[:, :, :3] + (1 - alpha) * region).astype("uint8")
        else:
            region[:] = image

    def _advance_round(self, detected_move):
        now = time.monotonic()
        with self._lock:
            if self._phase == "countdown" and now - self._round_started_at >= ROUND_SECONDS:
                if detected_move is None:
                    self._phase = "ready"
                    self._message = "Gesture not recognized. Try again."
                    return

                self._human_move = detected_move
                self._ai_move = MOVES[secrets.randbelow(len(MOVES))]
                self._winner = decide_winner(self._human_move, self._ai_move)
                if self._winner == "You win":
                    self._human_score += 1
                elif self._winner == "AI wins":
                    self._ai_score += 1
                self._phase = "result"
                self._result_started_at = now
                self._message = f"{self._winner}: {self._human_move} vs {self._ai_move}"

            elif self._phase == "result" and now - self._result_started_at >= RESULT_SECONDS:
                self._phase = "ready"
                self._human_move = None
                self._ai_move = None
                self._winner = None
                self._message = "Press Play round to begin"

    def recv(self, frame):
        image = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        results = self._hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detected_move = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self._draw.draw_landmarks(image, hand_landmarks, self._hand_connections)
                detected_move = self._get_move(hand_landmarks)

        self._advance_round(detected_move)
        height, width = image.shape[:2]
        state = self.snapshot()

        cv2.rectangle(image, (0, 0), (width, 78), (22, 28, 36), -1)
        cv2.putText(
            image,
            f"YOU {state['human_score']}   :   {state['ai_score']} AI",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (245, 245, 245),
            2,
        )

        if state["phase"] == "countdown":
            seconds_left = max(1, ROUND_SECONDS - int(time.monotonic() - self._round_started_at))
            cv2.putText(
                image,
                str(seconds_left),
                (width // 2 - 25, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (40, 220, 255),
                6,
            )

        if state["phase"] == "result":
            self._overlay_image(image, self._emoji_images[self._ai_move], 20, 92)
            self._overlay_image(image, self._emoji_images[self._human_move], width - 108, 92)
            cv2.putText(
                image,
                f"AI: {self._ai_move}  |  You: {self._human_move}",
                (18, height - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                self._winner,
                (width // 2 - 105, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (40, 220, 255),
                3,
            )
        elif state["phase"] == "ready":
            cv2.putText(
                image,
                state["message"],
                (18, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(page_title="Hand Gesture RPS", page_icon="✊", layout="centered")
st.title("Rock Paper Scissors")
st.caption("Play against the computer using your webcam.")


@st.fragment(run_every="300ms")
def game():
    context = webrtc_streamer(
        key="rock-paper-scissors",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=RockPaperScissorsProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    processor = context.video_processor
    if processor is None:
        st.info("Allow camera access to start playing.")
        return

    state = processor.snapshot()
    st.button(
        "Play round",
        type="primary",
        disabled=state["phase"] != "ready",
        on_click=processor.start_round,
    )
    score_col, ai_col = st.columns(2)
    score_col.metric("Your score", state["human_score"])
    ai_col.metric("AI score", state["ai_score"])
    if state["message"] and state["phase"] != "result":
        st.caption(state["message"])


game()
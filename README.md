\# ✊✋✌️ Rock Paper Scissors using Computer Vision



A real-time \*\*Rock Paper Scissors game\*\* built using \*\*Python, OpenCV, and MediaPipe\*\*, where a human player competes against an AI using \*\*hand gestures detected via webcam\*\*.



This project demonstrates practical usage of \*\*computer vision, hand landmark detection, and gesture recognition\*\*.



---



\## 🎯 Features



\- Real-time hand gesture recognition

\- AI opponent with random move generation

\- Countdown-based gameplay

\- Emoji visualization for moves

\- Score tracking

\- Smooth OpenCV UI

\- Works with a standard webcam



---



\## 🧠 Tech Stack



\- \*\*Python\*\*

\- \*\*OpenCV\*\*

\- \*\*MediaPipe\*\*

\- \*\*NumPy\*\*



---



\## 📁 Project Structure



computer-vision-rock-paper-scissors/

│

├── main.py

├── requirements.txt

├── README.md

├── emojis/

│ ├── rock.png

│ ├── paper.png

│ └── scissors.png





---



\## ⚙️ Installation \& Setup



\### 1️⃣ Clone the Repository

```bash

git clone https://github.com/your-username/computer-vision-rock-paper-scissors.git

cd computer-vision-rock-paper-scissors



2️⃣ Create Virtual Environment

python -m venv venv



Activate it:



Windows

venv\\Scripts\\activate

Mac / Linux

source venv/bin/activate



3️⃣ Install Dependencies

pip install -r requirements.txt



▶️ Run the Game

python main.py

Press q to quit the game.



🖐️ Hand Gestures

Gesture	Move

Fist	Rock

Two Fingers	Scissors

Open Palm	Paper



## Deploy with Streamlit Community Cloud

The Streamlit version uses the visitor's browser camera. The original `main.py` remains the desktop OpenCV version.

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Create an app from the repository, select `streamlit_app.py` as the main file, and choose Python 3.11 under Advanced settings.
4. Open the HTTPS app URL and allow camera access when prompted.

The deployment dependencies are listed in `requirements.txt`. Webcam access requires HTTPS and permission from the user. If video does not connect on a restrictive network, configure a TURN server and add its credentials to the `rtc_configuration` in `streamlit_app.py`.

To run the web app locally:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

🚀 Future Improvements

Two-player mode

Sound effects

Gesture accuracy improvement

Mobile/Web version



👩‍💻 Author

Ishita Garg

B.Tech Student | Data Science \& Machine Learning Enthusiast



⭐ If you like this project

Give it a ⭐ on GitHub 


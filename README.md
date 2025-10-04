# TrafficHawk-
🚦 AI Traffic Violation Detection System

An intelligent computer vision–based system that automatically detects common traffic violations from video feeds. This project was developed as part of the Institution’s Innovation Council (IIC) 2.0 Shortlisting Round submission by our team.
Our mission: to enhance road safety and assist traffic management authorities through real-time AI-powered monitoring.

🎯 The Problem

Manual traffic monitoring is inefficient, error-prone, and cannot provide continuous surveillance. As a result, numerous violations go unnoticed, compromising road discipline and safety.
Our system addresses this issue by automating traffic violation detection, enabling continuous, accurate, and unbiased monitoring.

✨ Key Features

✅ No Helmet Detection – Identifies riders without helmets using YOLOv8-based detection.
✅ Triple Riding Detection – Detects more than two people on a motorcycle.
✅ Red Light Jumping – Detects vehicles crossing the stop line when the signal is red.
✅ Evidence Generation – Captures violation images with timestamps for evidence.
✅ License Plate Recognition (in progress) – Extracts vehicle numbers for automatic challan generation.

🛠️ Tech Stack & Architecture

Programming Language: Python
Core Libraries: OpenCV, NumPy, TensorFlow / PyTorch
Detection Model: YOLOv8 (fine-tuned on custom datasets)
Tracking Algorithm: DeepSORT (for consistent object tracking across frames)

⚙️ How It Works

Input Source → Live CCTV feed or recorded video.

Detection → YOLOv8 identifies vehicles, riders, helmets, and traffic lights.

Violation Logic → Custom rules determine if a traffic rule has been broken.

Output → Snapshot, timestamp, and video overlay showing detected violations.

🚀 Getting Started
Prerequisites

Python 3.8 or above

pip (Python package installer)

Git

Installation
# Clone the repository
git clone https://github.com/durgesh-18101/traffichawk.git

# Navigate to project directory
cd Traffic Hawk

# Install dependencies
pip install -r requirements.txt

Model Setup

Download YOLOv8 weights (or your trained model) and place them inside the /weights directory.
(Add your download link here)

🖥️ Usage

Run detection on a video file or live webcam feed:

# For webcam
python detect.py --source 0

# For video file
python detect.py --source /path/to/video.mp4


Detected violations will be displayed live and saved as image evidence in the /output directory.

🖼️ Demo Snapshots

No Helmet Detection

Rider detected without a helmet (flagged with bounding box)

Triple Riding Detection

Motorcycle with three riders detected and logged

(You can add images or GIFs here to show system output)

👥 Team

Suryansh Seth – Project Lead / System Integration
Durgesh Sharma – Backend developer / AI Developer
Ujjwal Kansal -  UI/UX Design & Documentation

📜 License

This project is licensed under the MIT License.
See LICENSE.md
 for more details.

🙏 Acknowledgments

Special thanks to our mentors and the Institution’s Innovation Council (IIC) for guiding and inspiring us to create an AI-based solution for smarter, safer traffic management.

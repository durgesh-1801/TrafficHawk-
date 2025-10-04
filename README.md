# TrafficHawk-
🚦 AI Traffic Violation Detection System
An intelligent system designed to automatically detect common traffic violations from video feeds. This project was developed as our submission for the Institution's Innovation Council (IIC) 2.0 shortlisting round. Our goal is to enhance road safety and assist traffic management authorities using computer vision.
🎯 The Problem
Manual monitoring of traffic is inefficient, prone to human error, and cannot provide 24/7 coverage. This leads to a high number of un-penalized traffic violations, endangering public safety. Our solution aims to automate this process, making our roads safer and more disciplined.
✨ Key Features
This system can accurately detect several types of traffic violations in real-time:

⛑️ No Helmet Detection: Identifies motorcycle riders who are not wearing a helmet.

👨‍👨‍👧 Triple Riding Detection: Flags motorcycles carrying more than two people.

🚦 Red Light Jumping: Detects vehicles crossing the stop line when the traffic signal is red.

📸 Evidence Generation: Automatically captures a snapshot of the violation with a timestamp for evidence.

🔢 License Plate Recognition: (In-progress) Extracts the vehicle's license plate number for automated challan (ticket) generation.
🛠️ Tech Stack & Architecture
he project is built using a modern stack of computer vision and machine learning technologies.

Programming Language: Python

Core Libraries: OpenCV, NumPy, TensorFlow/PyTorch

Object Detection Model: We have utilized a pre-trained YOLOv8 (You Only Look Once) model, fine-tuned on custom datasets for high accuracy in detecting vehicles, riders, helmets, and traffic lights.

Tracking Algorithm: DeepSORT is used for tracking objects across frames to identify violations like red light jumping.

How It Works
Video Input: The system takes input from a live CCTV camera feed or a pre-recorded video file.

Frame Processing: Each frame is passed to the YOLOv8 model for object detection.

Violation Logic: Custom logic is applied to the detected objects and their positions to identify specific violations (e.g., a person on a motorcycle without a helmet).

Output: When a violation is detected, the system logs the event, saves an image of the infraction, and displays the violation on the output feed.
🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

Prerequisites
Python 3.8 or higher

pip (Python package installer)

Git
Installation:
1. Clone the repository:
   git clone https://github.com/durgesh-1801/traffichawk.git
2. Navigate to the project directory:
   cd traffichawk

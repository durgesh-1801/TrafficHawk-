# 🚦 AI Traffic Violation Detection  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)](https://opencv.org/)  
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red?logo=ultralytics)](https://github.com/ultralytics/ultralytics)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

An AI-powered system designed to automatically detect and log common **traffic violations** from live CCTV feeds or recorded videos. Built for the **Institution’s Innovation Council (IIC) 2.0 Shortlisting Round**, this project leverages computer vision to enhance **road safety** and support **traffic management authorities**.  

---

## ✨ Our Solution  

Traffic monitoring is mostly manual, error-prone, and inefficient.  
Our system solves this by using **deep learning models** to detect violations in real time and generate **evidence snapshots** with timestamps.  

Key violations detected:  
- ⛑️ **No Helmet Detection**  
- 👨‍👨‍👧 **Triple Riding Detection**  
- 🚦 **Red Light Jumping**  
- 📸 **Evidence Generation (Snapshots + Logs)**  
- 🔢 **License Plate Recognition (In-progress)**  

---

## 🎥 Prototype Showcase  

We implemented a working prototype to demonstrate real-world use cases:  
- Real-time detection on **webcam feeds**  
- Playback analysis on **video files**  
- Automatic snapshot storage for **violation reports**  

---

## ✅ Core Features Implemented  

- **YOLOv8 Object Detection** → Detects vehicles, helmets, riders, and signals  
- **DeepSORT Tracking** → Tracks objects across frames for violation logic  
- **Violation Logic Module** → Applies custom rules (helmet check, stop line, rider count)  
- **Evidence Generation** → Saves violation images with timestamp and label  
- **(Optional) LPR Module** → License Plate Recognition for automated challan  

---

## 🛠️ Tech Stack  

- **Language:** Python  
- **Libraries:** OpenCV, NumPy, TensorFlow / PyTorch  
- **Detection Model:** YOLOv8 (fine-tuned on custom dataset)  
- **Tracking:** DeepSORT  
- **Environment:** Jupyter / Command-line execution  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- pip  
- Git  



### Installation  
```bash
# Clone the repository
git clone https://github.com/your-username/traffichawk-.git

# Move into the project folder
cd traffic hawk-

# Install dependencies
pip install -r requirements.txt
# Run detection on webcam
python detect.py --source 0

## 🎥 Source Video

[Soure Video] (https://drive.google.com/file/d/1gKNJDTqP95fHPN80k970EFUkTRPlA_u_/view?usp=sharing)

---

```
## 👥 Team

| Name | Role | Responsibilities |
|------|------|------------------|
| **Suryansh Seth** | Project Lead / AI Developer | Model development, integration, and testing |
| **Durgesh Sharma** | Backend Engineer | API development, backend logic, and deployment |
| **Ujjwal Kansal** | UI/UX Designer | Interface design, documentation, and visualization |

## 📜 License  

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this project with attribution.  
See the [LICENSE](LICENSE) file for more details.  

**MIT License**

---

## 🙏 Acknowledgments  

We would like to thank:  

- **Institution’s Innovation Council (IIC)** – for providing the opportunity to innovate and present our project.  
- **Our Mentors and Faculty** – for continuous support, guidance, and valuable feedback.  
- **Open-Source Community** – for contributing libraries like YOLOv8, OpenCV, and TensorFlow, which made this project possible.  

> 🚀 *Together, we aim to make road safety smarter and more efficient through AI.*

---



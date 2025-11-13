# PerformaNet

A computer vision project that detects performative gestures - specifically when you're holding a cup/matcha for the fashion. Uses MediaPipe hand tracking to analyze hand gestures in real-time through your webcam.

## Description

PerformaNet uses computer vision and machine learning to detect when someone is holding a cup or matcha in their hand. The application analyzes hand landmarks in real-time using MediaPipe's hand tracking solution and determines if the hand gesture matches a "holding cup" pose based on finger positions and thumb placement.

## Features

- Real-time hand gesture detection using MediaPipe
- Visual feedback with hand landmark overlays
- Detection of performative cup/matcha holding gestures
- Screenshot capture functionality
- Mirror mode for natural interaction

## Requirements

- Python 3.7+
- Webcam/camera
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd PerformaNet
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure your webcam is connected and working
2. Run the application:
```bash
python main.py
```

3. The application will open a window showing your camera feed with hand tracking overlays
4. Hold your hand in a cup-holding gesture to see the detection in action
5. Controls:
   - `q` - Quit the application
   - `s` - Save a screenshot of the current frame

## How It Works

The application uses MediaPipe's hand tracking to identify 21 hand landmarks. It then analyzes the positions of these landmarks to determine if the hand is forming a cup-holding gesture by:
- Checking if fingers are curled (tips closer to palm)
- Verifying thumb position for gripping
- Detecting a C-shape or grip formation

When a performative gesture is detected, the application displays a yellow border and text overlay indicating the detection.

## Credits

Inspired by @aaronkh4n on TikTok

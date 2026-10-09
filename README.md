# Body Language Scoring for Interviews

A lightweight web application that evaluates interview performance using non-verbal cues captured from a webcam. By analyzing eye contact, head posture, and overall facial orientation, the app provides real-time feedback to help users improve presentation, confidence, and professionalism during interviews.

## Overview

This project uses computer vision and browser-based pose analysis to estimate how effectively a person communicates through body language. It is designed for interview practice, coaching, and self-assessment, making it easier to identify areas such as gaze stability, focus, and alignment with the camera.

The app processes live video input in the browser and uses MediaPipe Face Landmarker to detect facial landmarks, estimate head rotation, and score visual indicators relevant to interview presence.

## Key Features

- Real-time webcam analysis
- Eye contact scoring based on gaze direction and visual consistency
- Head posture assessment using facial orientation
- Overall interview readiness score
- Adaptive calibration for better scoring accuracy
- Instant feedback for posture and gaze correction
- Simple, browser-based interface with no backend required

## How It Works

1. The user enables webcam access from the browser.
2. The app captures live video frames.
3. MediaPipe detects facial landmarks and estimates head orientation.
4. The system evaluates:
   - eye movement and gaze stability
   - head position and alignment
   - facial orientation relative to the camera
5. A total score is generated and displayed in real time.

## Tech Stack

- HTML5
- CSS3
- JavaScript
- MediaPipe Tasks Vision
- Web APIs for camera access and browser rendering

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/AhryRaj/Bodylanguage_scoring.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Bodylanguage_scoring
   ```

3. Open `index.html` in a browser, or serve the project locally using a lightweight HTTP server.

Example:
```bash
python -m http.server 8000
```

Then visit:
```bash
http://localhost:8000
```

## Use Cases

- Mock interview preparation
- Career coaching and communication training
- Performance analysis for presentations
- Personal confidence and self-improvement practice

## Notes

This project is intended for educational and demonstration purposes and provides heuristic-based visual scoring rather than medical or professional evaluation. For best results, use adequate lighting and ensure the face remains clearly visible to the camera.

## License

This project is open for educational and personal use. Please review the repository license before commercial deployment.

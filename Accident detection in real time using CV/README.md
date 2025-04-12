# application for detecting accidents in real time from an aerial view (Bird's eye view)

This AI application is built to detect road accidents / vehicle accidents in real time using aerial cameras such as drone cameras ad visual input data.
This application is built on Python programmin language and uses OpenCV to detect potential accidents in video footage or live camera feeds. This system will analyze motion patterns, detect sudden changes, and identify potential collision events.

1. This accident detection system uses multiple computer vision techniques to identify potential accidents:

* Background Subtraction: Detects moving objects by separating foreground from background
* Optical Flow: Tracks motion between consecutive frames
* Motion Analysis: Monitors the intensity and patterns of motion
* Sudden Change Detection: Identifies abrupt changes that might indicate collisions

2. How the System Works

* Preprocessing: Each frame is resized, converted to grayscale, and blurred to reduce noise
* Motion Calculation: The system uses both background subtraction and optical flow to calculate motion levels
* Accident Detection: Potential accidents are identified when:
     * Overall motion exceeds a threshold
     * There's a sudden spike in motion (indicatin collision)
* Visualization: The system displays:
     * Real-time motion graph
     * Motion level metrics
     * Visual alerts when accidents are detected

To analyse video file:
system = AccidentDetectionSystem(video_source="path/to/your/video.mp4")
system.run()

To analyse in real time:
system = AccidentDetectionSystem(video_source=0)
system.run()

Adjusting sensitivity by changing thresholds:
system = AccidentDetectionSystem(
    motion_threshold=800,    # Lower = more sensitive
    sudden_motion_threshold=2000  # Lower = more sensitive
)


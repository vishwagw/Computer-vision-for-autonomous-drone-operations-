# This program can be used for both real time and recorded data (for testing) based detections.
# program codbase:
import cv2
import numpy as np
import time
from collections import deque

# building the Accident detection class:
class AccidentDetectionSystem:
    def __init__(self, video_source=0, buffer_size=30, motion_threshold=800, 
                 sudden_motion_threshold=2000, detection_cooldown=50):
        """
        Initialize the accident detection system.
        
        Parameters:
        - video_source: Camera index or video file path
        - buffer_size: Number of frames to keep in memory for analysis
        - motion_threshold: Threshold for motion detection
        - sudden_motion_threshold: Threshold for sudden motion (potential accident)
        - detection_cooldown: Frames to wait after detection before detecting again
        """
        self.video_source = video_source
        self.buffer_size = buffer_size
        self.motion_threshold = motion_threshold
        self.sudden_motion_threshold = sudden_motion_threshold
        self.detection_cooldown = detection_cooldown
        
        # Motion history
        self.motion_history = deque(maxlen=buffer_size)
        
        # Cooldown counter
        self.cooldown_counter = 0
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40, detectShadows=False)
        
        # Initialize optical flow parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7)
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Previous frame and points for optical flow
        self.prev_frame = None
        self.prev_points = None

    # Process a single frame and return annotated frame with accident detection
    def process_frame(self, frame):
        """Process a single frame and return annotated frame with accident detection"""
        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Create a copy of the original frame for drawing
        result_frame = frame.copy()
        
        # Calculate foreground mask
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Calculate motion metrics
        motion_level = self._calculate_motion(gray, fg_mask)
        self.motion_history.append(motion_level)
        
        # Check for accident only if not in cooldown
        detected = False
        if self.cooldown_counter <= 0:
            detected = self._detect_accident()
            if detected:
                # Start cooldown
                self.cooldown_counter = self.detection_cooldown
        else:
            self.cooldown_counter -= 1
                
        # Draw motion level graph
        self._draw_motion_graph(result_frame)
        
        # Add status text
        cv2.putText(result_frame, f"Motion: {motion_level}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add alert if accident detected
        if detected:
            cv2.putText(result_frame, "ACCIDENT DETECTED!", (120, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            # Draw red border
            cv2.rectangle(result_frame, (0, 0), (639, 479), (0, 0, 255), 10)
        
        return result_frame, detected
    
    # Calculate motion level using optical flow and foreground mask
    def _calculate_motion(self, gray, fg_mask):
        """Calculate motion level using optical flow and foreground mask"""
        motion_score = np.sum(fg_mask) / 255  # Basic motion from background subtraction
        
        # Calculate optical flow if we have previous frame
        if self.prev_frame is not None:
            # Find good features to track
            if self.prev_points is None or len(self.prev_points) < 10:
                self.prev_points = cv2.goodFeaturesToTrack(self.prev_frame, 
                                                         mask=None, 
                                                         **self.feature_params)
            
            if self.prev_points is not None and len(self.prev_points) > 0:
                # Calculate optical flow
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray, self.prev_points, None, **self.lk_params)
                
                # Select good points
                if next_points is not None:
                    good_new = next_points[status == 1]
                    good_old = self.prev_points[status == 1]
                    
                    # Calculate motion from optical flow
                    if len(good_new) > 0 and len(good_old) > 0:
                        flow_motion = np.sum(np.sqrt(
                            np.sum((good_new - good_old)**2, axis=1)))
                        motion_score += flow_motion
                    
                    # Update previous points
                    self.prev_points = good_new.reshape(-1, 1, 2)
            
        # Update previous frame
        self.prev_frame = gray.copy()
        
        return motion_score
    
    # Detect potential accidents based on motion history
    def _detect_accident(self):
        """Detect potential accidents based on motion history"""
        if len(self.motion_history) < self.buffer_size:
            return False
        
        # Check for sustained high motion
        if np.mean(self.motion_history) > self.motion_threshold:
            # Check for sudden spike in motion (indicating potential collision)
            for i in range(1, len(self.motion_history)):
                if (self.motion_history[i] - self.motion_history[i-1]) > self.sudden_motion_threshold:
                    return True
        
        return False
    
    # Draw motion history graph on the frame
    def _draw_motion_graph(self, frame):
        """Draw motion history graph on the frame"""
        if len(self.motion_history) < 2:
            return
        
        # Calculate graph dimensions
        graph_height = 100
        graph_width = 200
        x_offset = 420
        y_offset = 350
        
        # Draw graph background
        cv2.rectangle(frame, (x_offset, y_offset), 
                     (x_offset + graph_width, y_offset + graph_height), 
                     (0, 0, 0), -1)
        
        # Draw threshold line
        threshold_y = y_offset + graph_height - int(
            (self.motion_threshold / max(max(self.motion_history), 1)) * graph_height)
        cv2.line(frame, (x_offset, threshold_y), 
                (x_offset + graph_width, threshold_y), (0, 255, 255), 1)
        
        # Draw motion graph
        max_motion = max(max(self.motion_history), 1)
        points = []
        
        for i, motion in enumerate(self.motion_history):
            x = x_offset + int((i / len(self.motion_history)) * graph_width)
            y = y_offset + graph_height - int((motion / max_motion) * graph_height)
            points.append((x, y))
        
        # Draw lines between points
        for i in range(1, len(points)):
            color = (0, 255, 0)
            # If motion exceeds threshold, change color to yellow
            if self.motion_history[i] > self.motion_threshold:
                color = (0, 255, 255)
            # If sudden increase, change color to red
            if i > 0 and (self.motion_history[i] - self.motion_history[i-1]) > self.sudden_motion_threshold:
                color = (0, 0, 255)
            
            cv2.line(frame, points[i-1], points[i], color, 2)
        
        # Label the graph
        cv2.putText(frame, "Motion History", (x_offset, y_offset - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    # Run the accident detection system on video source
    def run(self):
        """Run the accident detection system on video source"""
        # Open video capture
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        print("Accident Detection System Started")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream.")
                break
            
            # Process the frame
            result_frame, detected = self.process_frame(frame)
            
            # Display the result
            cv2.imshow("Accident Detection", result_frame)
            
            # Log accident detection
            if detected:
                print(f"Potential accident detected at {time.strftime('%H:%M:%S')}")
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

# initialzing the program:
if __name__ == "__main__":
    # For webcam use video_source=0, or provide path to video file
    # Adjust thresholds based on testing
    system = AccidentDetectionSystem(
        video_source='./drone accident footage 2.mp4',  # Use 0 for webcam or provide video path
        motion_threshold=1000,
        sudden_motion_threshold=3000
    )
    system.run()

import cv2
import numpy as np
import os
import shutil
import time


# Path to save intermediate objects
shutil.rmtree("processed_frames", ignore_errors=True)
output_folder = "processed_frames"
os.makedirs(output_folder, exist_ok=True)

THRESHOLD = 120

# Process all frames in the "frames" folder
frames_folder = "frames"
for image_file in sorted(os.listdir(frames_folder)):
    # Read frame
    frame = cv2.imread(os.path.join(frames_folder, image_file))
    if frame is None:
        print(f"Failed to read the frame: {image_file}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "grayscale.jpg"), gray)

    # Threshold the image to isolate the ball
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(output_folder, "thresholded.jpg"), thresh)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the enclosing circle or bounding box for the largest object
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        if radius > 5:  # Ignore small objects
            # Draw the object on the original frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"Position: ({int(x)}, {int(y)})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the processed frame
    cv2.imshow('Processed Frame', frame)
    
    # Wait for 1 second and check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1)

cv2.destroyAllWindows()

# Save the final processed frame
cv2.imwrite(os.path.join(output_folder, "final_frame.jpg"), frame)

print(f"Processed images saved to '{output_folder}'")


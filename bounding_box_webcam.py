import cv2
import os
import time

# Create a video capture object
cap = cv2.VideoCapture(0)

# video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


# Define the codec for the output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# VideoWriter object to save the compressed video
out = cv2.VideoWriter('bounding_box_webcam.mp4', fourcc, 30, (width, height))

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Variables for FPS calculation
frame_count = 0
start_time = time.time()

while True:
    # Read the frame from the video
    ret, frame = cap.read()

    # If the frame is not retrieved, break out of the loop
    if not ret:
        break

    # Applying Gaussian blur for more accuracy of background subtraction
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply background subtraction to the blurred frame
    fg_mask = bg_subtractor.apply(blurred_frame)

    # Perform thresholding to obtain a binary image
    _, binary_image = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour
    for contour in contours:
        # Calculate the contour area
        area = cv2.contourArea(contour)

        # Filter contours based on area
        if area > 1000:  # Adjust the threshold as needed
            # Draw a bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            #bounding rectangle is the smallest rectange that
            #encloses the contour
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #(x,y) are the coordinates of the top left corner
            #of the bounding box

    # Display the frame with bounding boxes
    cv2.imshow('Bounding Boxes', frame)

    out.write(frame)
    # Wait for a key press and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count=frame_count+1

# Release the video capture object
cap.release()

# Calculate FPS
end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time
elapsed_time = time.time() - start_time

# Print the calculated FPS value
print("FPS:{:.2f}".format(fps))

# Get the file size
file_path = 'output_video.mp4'
file_size_bytes = os.path.getsize(file_path)
file_size_mb= file_size_bytes/1048576
print("File size: {:.2f} MB".format(file_size_mb))

#Duration in MB
print("Duration of video(in seconds): {:.2f} s". format(elapsed_time))
duration_mb = file_size_mb / fps
print("Duration :{:.2f} MB".format(duration_mb))

#Resolution (in MB)
resolution_mb = (width * height * file_size_mb) / (1024 * 1024)
print("Resolution of video: {:.2f} MB".format(resolution_mb))



# Destroy any remaining windows
cv2.destroyAllWindows()
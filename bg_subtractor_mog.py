import cv2
import os
cap = cv2.VideoCapture('assets/walking.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# Get the  original file size
file_path = 'assets/walking.mp4'
file_size_bytes = os.path.getsize(file_path)
file_size_mb = file_size_bytes / 1048576

print("Video size: {:.2f} MB".format(file_size_mb))

# Get the original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("fps of video: {:.2f}".format(fps))

# Resolution of the original video
vid_resolution= (width * height * file_size_mb) / (1024 * 1024)
print("Resolution of original video: {:.2f} MB".format(vid_resolution))

# Define the codec for the output video file
fourcc = cv2.VideoWriter_fourcc(*'h264')

# VideoWriter object to save the  video
out = cv2.VideoWriter('bgSubMOG.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    #returns a binary foreground mask image (fgmask)
    #where white pixels represent the foreground objects.

    cv2.imshow('Frame', frame)   #original frame
    cv2.imshow('FG MASK Frame', fgmask)    #foreground mask frame

    out.write(fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
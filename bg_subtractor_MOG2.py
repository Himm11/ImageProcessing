import cv2
cap = cv2.VideoCapture('walking.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Get the original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for the output video file
fourcc = cv2.VideoWriter_fourcc(*'h264')

# VideoWriter object to save the  video
out = cv2.VideoWriter('bgSubMOG2.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG MASK Frame', fgmask)

    out.write(fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
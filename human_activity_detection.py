import cv2

# Load the pre-trained human detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open a video capture object (you can also use cv2.VideoCapture(0) for webcam)
cap = cv2.VideoCapture('aaa.mp4')

# Load the action labels from Action.txt
with open('Actions.txt', 'r') as file:
    action_labels = file.read().splitlines()

frame_number = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Detect humans in the frame
    humans, _ = hog.detectMultiScale(frame)

    # Draw bounding boxes around detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the action label on the frame
    action_label = action_labels[frame_number]
    cv2.putText(frame, f"Action: {action_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Human Activity Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

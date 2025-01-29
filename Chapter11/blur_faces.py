import cv2
import numpy as np

def blur_face_and_plates(image_path, save_path):
    """
    A simple function to detect and blur faces and license plates in an image
    """
    # Read the image
    image = cv2.imread(image_path)

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load license plate detection model
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Blur each face
    for (x, y, w, h) in faces:
        # Extract the region of interest (face)
        roi = image[y:y+h, x:x+w]
        # Apply blur
        blurred = cv2.GaussianBlur(roi, (23, 23), 30)
        # Put the blurred region back into the image
        image[y:y+h, x:x+w] = blurred

    # Detect license plates
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)

    # Blur each license plate
    for (x, y, w, h) in plates:
        roi = image[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (23, 23), 30)
        image[y:y+h, x:x+w] = blurred

    # Save the result
    cv2.imwrite(save_path, image)

    return image

# For processing video
def process_video(input_video, output_video):
    """
    Process a video file to blur faces and license plates
    """
    # Open the video
    cap = cv2.VideoCapture(input_video)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer
    out = cv2.VideoWriter(output_video,
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (width, height))

    # Load the cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect and blur faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (23, 23), 30)
            frame[y:y+h, x:x+w] = blurred

        # Detect and blur plates
        plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in plates:
            roi = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (23, 23), 30)
            frame[y:y+h, x:x+w] = blurred

        # Write the frame
        out.write(frame)

    # Release everything
    cap.release()
    out.release()

# Example usage
if __name__ == "__main__":
    # For images
    blur_face_and_plates('Am.jpeg', 'anonymized_image.jpg')

    # For videos
    #process_video('input_video.mp4', 'anonymized_video.mp4')

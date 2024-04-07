import cv2
import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace
import threading

# Function to perform face verification
def verify_face(reference_img, frame):
    try:
        result = DeepFace.verify(reference_img, frame)['verified']
    except Exception as e:
        st.error(f"Error during face verification: {e}")
        result = False
    return result

def main():
    st.title("Real-time Face Matching App")

    # Upload reference image
    st.subheader("Upload Reference Image:")
    reference_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)  # Use CAP_ANY as the backend
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()

        if ret:
            # Display webcam frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, channels="RGB", caption="Webcam Stream", use_column_width=True)

            # Perform face detection and verification
            if reference_image is not None:
                reference_img = Image.open(reference_image)
                aligned_reference_img = preprocess_image(reference_img)  # Preprocess reference image

                try:
                    result = verify_face(aligned_reference_img, frame.copy())
                    if result:
                        st.success("MATCH! Face Detected.")
                    else:
                        st.warning("NO MATCH! Face Not Detected.")
                except Exception as e:
                    st.error(f"Error during face verification: {e}")

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image):
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Resize image to a fixed size (e.g., 224x224) for face verification
    resized_image = cv2.resize(image_np, (224, 224))
    
    # Convert BGR image to RGB (if necessary)
    if len(resized_image.shape) > 2 and resized_image.shape[2] == 3:
        # OpenCV uses BGR by default, convert to RGB
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    return resized_image

if __name__ == "__main__":
    main()

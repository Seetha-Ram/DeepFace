import cv2
import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace
import tempfile
import os

# Function to perform face verification
def verify_face(reference_img, frame):
    try:
        result = DeepFace.verify(reference_img, frame)['verified']
    except Exception as e:
        st.error(f"Error during face verification: {e}")
        result = False
    return result

def main():
    st.title("Face Matching App from Uploaded Video")

    # Upload reference image
    st.subheader("Upload Reference Image:")
    reference_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Upload video file
    st.subheader("Upload Video File:")
    video_file = st.file_uploader("Choose a video file...", type=["mp4"])

    if video_file is not None:
        # Save video file to a temporary location
        temp_video_path = save_uploaded_file(video_file)

        if temp_video_path is not None:
            # Read video file as OpenCV VideoCapture object
            video_cap = cv2.VideoCapture(temp_video_path)

            # Loop through video frames
            while True:
                ret, frame = video_cap.read()

                if not ret:
                    break  # Break the loop when the video ends

                # Display video frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame, channels="RGB", caption="Uploaded Video", use_column_width=True)

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

            video_cap.release()

def save_uploaded_file(uploaded_file):
    # Save the uploaded file to a temporary directory and return the file path
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_file_path

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

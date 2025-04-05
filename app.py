import gradio as gr
import face_recognition
import cv2
import os
from pathlib import Path
import numpy as np

# Folder for storing uploaded images and output cropped faces
UPLOAD_FOLDER = "uploaded_images"
OUTPUT_FOLDER = "output_faces"

# Create necessary folders if they do not exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Function to detect and crop faces from images using face_recognition
def detect_and_crop_faces(image):
    # Convert the image to RGB (Gradio sends images as numpy arrays)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image using face_recognition
    face_locations = face_recognition.face_locations(image_rgb)
    
    # If no faces found, return None
    if len(face_locations) == 0:
        return None, None

    cropped_faces = []
    output_images = []

    # Crop each face found and draw a rectangle on the original image
    for face_location in face_locations:
        top, right, bottom, left = face_location
        # Crop the face from the image
        cropped_face = image[top:bottom, left:right]
        
        # Save cropped face to the output folder
        face_filename = f"face_{os.path.basename(image)}"
        face_path = os.path.join(OUTPUT_FOLDER, face_filename)
        cv2.imwrite(face_path, cropped_face)
        cropped_faces.append(face_path)
        
        # Draw a rectangle around the face on the original image
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

    # Return the processed image with faces detected and the cropped face images
    output_image_path = os.path.join(OUTPUT_FOLDER, f"detected_{os.path.basename(image)}")
    cv2.imwrite(output_image_path, image)
    
    return image, cropped_faces


# Define the Gradio interface
def gradio_interface(image):
    processed_image, cropped_faces = detect_and_crop_faces(image)
    
    if processed_image is None:
        return "No faces detected", None

    return processed_image, cropped_faces


# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.inputs.Image(type="numpy"),  # Accept image as numpy array
    outputs=[gr.outputs.Image(type="numpy"), gr.outputs.JSON()],
    title="Face Recognition",
    description="Upload an image and get the detected faces and cropped faces.",
)

# Launch the Gradio app
iface.launch()

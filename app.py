from flask import Flask, request, jsonify
import face_recognition
import cv2
import os
from pathlib import Path

app = Flask(__name__)

# Folder for storing uploaded images and output cropped faces
UPLOAD_FOLDER = "uploaded_images"
OUTPUT_FOLDER = "output_faces"

# Create necessary folders if they do not exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Function to detect and crop faces from images using face_recognition
def detect_and_crop_faces(image_path: str):
    # Load image using face_recognition
    image = face_recognition.load_image_file(image_path)
    
    # Detect faces in the image
    face_locations = face_recognition.face_locations(image)
    
    # If no faces found, return None
    if len(face_locations) == 0:
        return None, None

    # Open the image using OpenCV for cropping and saving
    img = cv2.imread(image_path)
    
    cropped_faces = []
    output_images = []

    # Crop each face found
    for face_location in face_locations:
        top, right, bottom, left = face_location
        # Crop the face from the image
        cropped_face = img[top:bottom, left:right]
        
        # Save cropped face to the output folder
        face_filename = f"face_{os.path.basename(image_path)}"
        face_path = os.path.join(OUTPUT_FOLDER, face_filename)
        cv2.imwrite(face_path, cropped_face)
        cropped_faces.append(face_path)
        
        # Mark the face on the original image (draw a rectangle around the face)
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

    # Save the output image with faces detected
    output_image_path = os.path.join(OUTPUT_FOLDER, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(output_image_path, img)
    output_images.append(output_image_path)

    return cropped_faces, output_images


@app.route("/upload-images/", methods=["POST"])
def upload_images():
    # Get the folder parameter from the request
    folder = request.form.get("folder")
    input_folder = Path(folder)
    
    if not input_folder.exists():
        return jsonify({"error": "Folder not found"}), 404
    
    result = []

    # Process all .jpg files in the folder
    for image_file in input_folder.glob("*.jpg"):
        cropped_faces, output_images = detect_and_crop_faces(str(image_file))
        if cropped_faces and output_images:
            result.append({
                "cropped_face": cropped_faces,
                "matched_images": output_images
            })
    
    if not result:
        return jsonify({"error": "No faces detected in any images."}), 404
    
    return jsonify({"results": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

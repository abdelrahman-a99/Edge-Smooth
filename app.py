from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageFilter
import os

from Smoothing import denoise_image
from Edge_detection import Edge_Detection
from Negative_Transformation import negative_transform_color
from Blur import blur_image_cnn

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads/"
PROCESSED_FOLDER = "static/processed/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files or "algorithm" not in request.form:
        return "No file or algorithm selected"

    file = request.files["file"]
    algorithm = request.form["algorithm"]

    if file.filename == "":
        return "No selected file"

    if file:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # img = Image.open(file_path)
        # filename = os.path.basename(file.filename)
        # img = denoise_image(img)
        # img.save(processed_path)

        # Process the image
        # processed_image = Edge_Detection(file_path)
        # if processed_image is None:
            # return "Error processing image"

        # Save the processed image
        # processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
        # processed_image.save(processed_path)

        # Process the image using denoise_image function
        # processed_image = denoise_image(file_path)
        # if processed_image is None:
            # return "Error processing image"

        # Save the processed image
        # processed_path = os.path.join(PROCESSED_FOLDER, file.filename)
        # processed_image.save(processed_path)

        # return send_from_directory(PROCESSED_FOLDER, file.filename)

        processed_path = os.path.join(PROCESSED_FOLDER, file.filename)

        if algorithm == "edge_detection":
            processed_image = Edge_Detection(file_path)
            if processed_image is None:
                return "Error processing image"
            processed_image.save(processed_path)

        elif algorithm == "smoothing":
            denoised_image_path = denoise_image(file_path, processed_path)
            if denoised_image_path is None:
                return "Error processing image"

        elif algorithm == "cnn_blur":
            kernel_size = 10
            try:
                blur_image_cnn(file_path, kernel_size, processed_path)
            except ValueError as e:
                return str(e)
        
        elif algorithm == "negative_transform":
            # Placeholder for the negative transform algorithm
            img = Image.open(file_path).convert("L")
            negative_img = Image.eval(img, lambda x: 255 - x)
            negative_img.save(processed_path)

        elif algorithm == "negative_transform_color":
            # Handle negative transform for color images
            negative_transform_color(file_path, processed_path)

        # denoised_image_path = denoise_image(file_path, processed_path)
        # if denoised_image_path is None:
        #     return "Error processing image"

        return send_from_directory(PROCESSED_FOLDER, file.filename)

if __name__ == "__main__":
    app.run(debug=True)

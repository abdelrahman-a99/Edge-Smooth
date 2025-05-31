from flask import (Flask, render_template, request, send_from_directory, flash, redirect, url_for,)
from PIL import Image
import os
import logging
from werkzeug.utils import secure_filename
from typing import Optional, Tuple

from Smoothing import denoise_image
from Edge_detection import Edge_Detection
from Negative_Transformation import negative_transform_color
from Blur import blur_image_cnn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

# Configuration
UPLOAD_FOLDER = "static/uploads/"
PROCESSED_FOLDER = "static/processed/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(file_path: str) -> Tuple[bool, Optional[str]]:
    """Validate if the file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, str(e)


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and processing."""
    try:
        # Check if file and algorithm are present
        if "file" not in request.files or "algorithm" not in request.form:
            flash("No file or algorithm selected")
            return redirect(url_for("index"))

        file = request.files["file"]
        algorithm = request.form["algorithm"]

        # Check if file is selected
        if file.filename == "":
            flash("No selected file")
            return redirect(url_for("index"))

        # Validate file
        if not allowed_file(file.filename):
            flash("File type not allowed")
            return redirect(url_for("index"))

        # Secure the filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Validate image
        is_valid, error = validate_image(file_path)
        if not is_valid:
            flash(f"Invalid image file: {error}")
            return redirect(url_for("index"))

        # Process the image
        processed_path = os.path.join(PROCESSED_FOLDER, filename)

        try:
            if algorithm == "edge_detection":
                processed_image = Edge_Detection(file_path)
                if processed_image is None:
                    raise ValueError("Edge detection failed")
                processed_image.save(processed_path)

            elif algorithm == "smoothing":
                denoised_image_path = denoise_image(file_path, processed_path)
                if denoised_image_path is None:
                    raise ValueError("Smoothing failed")

            elif algorithm == "cnn_blur":
                kernel_size = 10
                result = blur_image_cnn(file_path, kernel_size, processed_path)
                if result is None:
                    raise ValueError("Blur failed")

            elif algorithm == "negative_transform":
                img = Image.open(file_path).convert("L")
                negative_img = Image.eval(img, lambda x: 255 - x)
                negative_img.save(processed_path)

            elif algorithm == "negative_transform_color":
                result = negative_transform_color(file_path, processed_path)
                if result is None:
                    raise ValueError("Negative transformation failed")

            else:
                flash("Invalid algorithm selected")
                return redirect(url_for("index"))

            logger.info(f"Successfully processed image with {algorithm}")
            return send_from_directory(PROCESSED_FOLDER, filename)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            flash(f"Error processing image: {str(e)}")
            return redirect(url_for("index"))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        flash("An unexpected error occurred")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)

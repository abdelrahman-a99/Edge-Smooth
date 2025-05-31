from flask import (Flask, render_template, request, send_from_directory, jsonify)
from PIL import Image
import os
import logging
from werkzeug.utils import secure_filename
from typing import Optional, Tuple

from app.utils.Smoothing import denoise_image
from app.utils.Edge_detection import Edge_Detection
from app.utils.Negative_Transformation import negative_transform_color
from app.utils.Blur import blur_image_cnn

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Get the absolute path to the app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(APP_DIR, 'static'), static_url_path='/static')
app.secret_key = os.urandom(24)  # Required for flash messages

# Configuration
UPLOAD_FOLDER = os.path.join(APP_DIR, "static", "uploads")
PROCESSED_FOLDER = os.path.join(APP_DIR, "static", "processed")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Log directory paths
logger.info(f"APP_DIR: {APP_DIR}")
logger.info(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logger.info(f"PROCESSED_FOLDER: {PROCESSED_FOLDER}")

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
            logger.error("No file or algorithm selected")
            return jsonify({"error": "No file or algorithm selected"}), 400

        file = request.files["file"]
        algorithm = request.form["algorithm"]
        logger.info(f"Processing request - Algorithm: {algorithm}, Filename: {file.filename}")

        # Check if file is selected
        if file.filename == "":
            logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        # Validate file
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "File type not allowed"}), 400

        # Secure the filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        logger.info(f"File saved to: {file_path}")

        # Validate image
        is_valid, error = validate_image(file_path)
        if not is_valid:
            logger.error(f"Invalid image file: {error}")
            return jsonify({"error": f"Invalid image file: {error}"}), 400

        # Process the image
        processed_path = os.path.join(PROCESSED_FOLDER, filename)
        logger.info(f"Processing image - Input: {file_path}, Output: {processed_path}")

        try:
            if algorithm == "edge_detection":
                processed_image = Edge_Detection(file_path)
                if processed_image is None:
                    raise ValueError("Edge detection failed")
                processed_image.save(processed_path)
                logger.info(f"Saved processed image to: {processed_path}")

            elif algorithm == "smoothing":
                logger.info("Starting smoothing process")
                denoised_image_path = denoise_image(file_path, processed_path)
                if denoised_image_path is None:
                    logger.error("Smoothing failed - no output path returned")
                    raise ValueError("Smoothing failed - no output path returned")
                logger.info(f"Saved processed image to: {processed_path}")

            elif algorithm == "cnn_blur":
                kernel_size = 10
                result = blur_image_cnn(file_path, kernel_size, processed_path)
                if result is None:
                    raise ValueError("Blur failed")
                logger.info(f"Saved processed image to: {processed_path}")

            elif algorithm == "negative_transform":
                img = Image.open(file_path).convert("L")
                negative_img = Image.eval(img, lambda x: 255 - x)
                negative_img.save(processed_path)
                logger.info(f"Saved processed image to: {processed_path}")

            elif algorithm == "negative_transform_color":
                result = negative_transform_color(file_path, processed_path)
                if result is None:
                    raise ValueError("Negative transformation failed")
                logger.info(f"Saved processed image to: {processed_path}")

            else:
                logger.error(f"Invalid algorithm selected: {algorithm}")
                return jsonify({"error": "Invalid algorithm selected"}), 400

            logger.info(f"Successfully processed image with {algorithm}")
            return jsonify({"success": True, "filename": filename})

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_from_directory(os.path.join("app", "static", "uploads"), filename)


@app.route("/static/processed/<filename>")
def processed_file(filename):
    """Serve processed images."""
    try:
        return send_from_directory(PROCESSED_FOLDER, filename)
    except Exception as e:
        logger.error(f"Error serving processed file {filename}: {str(e)}")
        return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(debug=True) 
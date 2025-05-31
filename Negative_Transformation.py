import cv2
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def negative_transform_color(input_path: str, output_path: str) -> Optional[str]:
    """
    Convert an image to its negative using vectorized operations.

    Args:
        input_path (str): Path to the input image
        output_path (str): Path where the negative image will be saved

    Returns:
        Optional[str]: Path to the saved negative image if successful, None otherwise

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the image cannot be read or processed
    """
    try:
        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not read image from {input_path}")

        # Convert to negative using vectorized operation
        negative_img = 255 - img

        # Save the negative image
        success = cv2.imwrite(output_path, negative_img)
        if not success:
            raise ValueError(f"Could not save image to {output_path}")

        logger.info(f"Successfully created negative image at {output_path}")
        return output_path

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return None
    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

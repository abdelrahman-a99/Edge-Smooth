import numpy as np
import cv2
from PIL import Image
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pad_image(image: np.ndarray, padding: int = 1) -> np.ndarray:
    """
    Add padding to the image to maintain dimensions after convolution.

    Args:
        image (np.ndarray): Input image
        padding (int): Padding size

    Returns:
        np.ndarray: Padded image
    """
    return np.pad(image, padding, mode="reflect")


def Edge_Detection(path: str, threshold: float = 0.0) -> Optional[Image.Image]:
    """
    Perform edge detection using Sobel operators.

    Args:
        path (str): Path to the input image
        threshold (float): Threshold value for edge detection (0-255)

    Returns:
        Optional[Image.Image]: Processed image with detected edges, or None if error occurs

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the image cannot be read or processed
    """
    try:
        # Read and validate image
        Fence_Gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if Fence_Gray is None:
            raise ValueError(f"Could not read image from {path}")

        # Add padding to maintain dimensions
        padded_image = pad_image(Fence_Gray)

        # Sobel operators
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Horizontal
        Ky = Kx.T  # Vertical

        # Initialize output arrays
        height, width = Fence_Gray.shape
        Fence_Gray_Edges1x = np.zeros((height, width))  # Horizontal edges
        Fence_Gray_Edges1y = np.zeros((height, width))  # Vertical edges
        Fence_Gray_Edges1 = np.zeros((height, width))  # Combined edges

        # Apply convolution
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                window = padded_image[i - 1 : i + 2, j - 1 : j + 2]
                Fence_Gray_Edges1x[i - 1, j - 1] = np.maximum(
                    np.sum(np.multiply(window, Kx)), 0
                )
                Fence_Gray_Edges1y[i - 1, j - 1] = np.maximum(
                    np.sum(np.multiply(window, Ky)), 0
                )
                Fence_Gray_Edges1[i - 1, j - 1] = np.sqrt(
                    Fence_Gray_Edges1x[i - 1, j - 1] ** 2
                    + Fence_Gray_Edges1y[i - 1, j - 1] ** 2
                )

        # Apply threshold
        if threshold > 0:
            Fence_Gray_Edges1 = np.where(
                Fence_Gray_Edges1 > threshold, Fence_Gray_Edges1, 0
            )

        # Normalize to 0-255 range
        Fence_Gray_Edges1 = cv2.normalize(
            Fence_Gray_Edges1, None, 0, 255, cv2.NORM_MINMAX
        )

        # Convert to PIL Image
        final_image = Image.fromarray(Fence_Gray_Edges1.astype(np.uint8))

        logger.info(f"Successfully processed image with edge detection")
        return final_image

    except FileNotFoundError:
        logger.error(f"Input file not found: {path}")
        return None
    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

import numpy as np
import tensorflow as tf
import cv2
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_kernel_size(kernel_size: int) -> int:
    """
    Validate and adjust kernel size to ensure it's odd.

    Args:
        kernel_size (int): Input kernel size

    Returns:
        int: Validated kernel size (odd number)

    Raises:
        ValueError: If kernel size is less than 3
    """
    if kernel_size < 3:
        raise ValueError("Kernel size must be at least 3")
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def blur_image_cnn(
    input_image_path: str, kernel_size: int, output_image_path: str
) -> Optional[str]:
    """
    Apply CNN-based blur effect to an image.

    Args:
        input_image_path (str): Path to the input image
        kernel_size (int): Size of the blur kernel (must be odd)
        output_image_path (str): Path where the blurred image will be saved

    Returns:
        Optional[str]: Path to the saved blurred image if successful, None otherwise

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the image cannot be read or processed
    """
    try:
        # Validate kernel size
        kernel_size = validate_kernel_size(kernel_size)

        # Load and validate image
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError(f"Could not read image from {input_image_path}")

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create and normalize the kernel
        kernel = np.ones((kernel_size, kernel_size, 1, 1), np.float32) / (
            kernel_size**2
        )
        kernel = np.repeat(kernel, 3, axis=2)  # Repeat for 3 input channels (RGB)
        kernel = np.repeat(kernel, 3, axis=3)  # Repeat for 3 output channels (RGB)

        # Convert image to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Expand dimensions for batch processing
        image = np.expand_dims(image, axis=0)

        # Create the CNN model
        input_shape = (None, None, 3)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=3,
                    kernel_size=kernel_size,
                    padding="same",
                    use_bias=False,
                    kernel_initializer=tf.constant_initializer(kernel),
                    input_shape=input_shape,
                )
            ]
        )

        # Compile the model
        model.compile(optimizer="adam", loss="mean_squared_error")

        # Apply blur
        blurred_image = model.predict(image)

        # Process output
        blurred_image = np.squeeze(blurred_image, axis=0)
        blurred_image = np.clip(blurred_image * 255, 0, 255).astype(np.uint8)

        # Convert back to BGR and save
        blurred_image_bgr = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(output_image_path, blurred_image_bgr)

        if not success:
            raise ValueError(f"Could not save image to {output_image_path}")

        logger.info(f"Successfully created blurred image at {output_image_path}")
        return output_image_path

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_image_path}")
        return None
    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
from tqdm.keras import TqdmCallback
import logging
from typing import Optional
import os
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Configure TensorFlow to use CPU only and limit memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU configuration successful. Found {len(gpus)} GPU(s)")
    else:
        logger.info("No GPU found, using CPU")
except Exception as e:
    logger.warning(f"GPU configuration failed: {str(e)}")

def validate_parameters(num_iters: int, init_lr: float, sig: float) -> tuple:
    """
    Validate and adjust parameters for the smoothing algorithm.

    Args:
        num_iters (int): Number of iterations
        init_lr (float): Initial learning rate
        sig (float): Noise level

    Returns:
        tuple: Validated parameters (num_iters, init_lr, sig)

    Raises:
        ValueError: If parameters are invalid
    """
    try:
        if num_iters < 100:
            raise ValueError("Number of iterations must be at least 100")
        if init_lr <= 0:
            raise ValueError("Learning rate must be positive")
        if sig <= 0:
            raise ValueError("Noise level must be positive")
        return num_iters, init_lr, sig
    except Exception as e:
        logger.error(f"Parameter validation failed: {str(e)}")
        raise

def denoise_image(input_image_path: str, output_image_path: str, num_iters: int = 200, init_lr: float = 0.01, sig: float = 30) -> Optional[str]:
    """
    Apply Deep Image Prior based denoising to an image.

    Args:
        input_image_path (str): Path to the input image
        output_image_path (str): Path where the denoised image will be saved
        num_iters (int): Number of iterations for the optimization
        init_lr (float): Initial learning rate
        sig (float): Noise level

    Returns:
        Optional[str]: Path to the saved denoised image if successful, None otherwise
    """
    try:
        logger.info(f"Starting denoising process for {input_image_path}")
        logger.info(f"Output path: {output_image_path}")
        
        # Check if input file exists
        if not os.path.exists(input_image_path):
            logger.error(f"Input file not found: {input_image_path}")
            raise FileNotFoundError(f"Input file not found: {input_image_path}")

        # Validate parameters
        try:
            num_iters, init_lr, sig = validate_parameters(num_iters, init_lr, sig)
            logger.info(f"Parameters validated: num_iters={num_iters}, init_lr={init_lr}, sig={sig}")
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return None

        # Function to add noise to an image
        def get_noisy_img(img, sig=30):
            try:
                sigma = sig / 255.
                noise = np.random.normal(scale=sigma, size=img.shape)
                img_noisy = np.clip(img + noise, 0, 1).astype(np.float32)
                return img_noisy
            except Exception as e:
                logger.error(f"Error adding noise: {str(e)}")
                raise

        # Function to load an image from a file path
        def load_image(image_path):
            try:
                logger.info(f"Loading image from {image_path}")
                img = imread(image_path).astype(np.float32) / 255.
                # Ensure image is RGB
                if len(img.shape) == 2:  # Grayscale
                    logger.info("Converting grayscale image to RGB")
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 4:  # RGBA
                    logger.info("Converting RGBA image to RGB")
                    img = img[..., :3]
                logger.info(f"Image loaded successfully with shape {img.shape}")
                return img
            except Exception as e:
                logger.error(f"Error loading image: {str(e)}")
                raise ValueError(f"Error loading image: {str(e)}")

        # Simplified Deep Image Prior model definition
        def deep_image_prior(input_shape, noise_reg=None, layers=(32, 64, 64, 32), kernel_size=3):
            try:
                logger.info(f"Creating model with input shape {input_shape}")
                def norm_and_active(x):
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.LeakyReLU()(x)
                    return x

                inputs = tf.keras.Input(shape=input_shape)
                x = inputs
                if noise_reg:
                    x = tf.keras.layers.GaussianNoise(noise_reg['stddev'])(x)

                # Simplified architecture
                x = norm_and_active(tf.keras.layers.Conv2D(layers[0], kernel_size, padding='same')(x))
                x = norm_and_active(tf.keras.layers.Conv2D(layers[1], kernel_size, padding='same')(x))
                x = norm_and_active(tf.keras.layers.Conv2D(layers[2], kernel_size, padding='same')(x))
                x = norm_and_active(tf.keras.layers.Conv2D(layers[3], kernel_size, padding='same')(x))
                x = tf.keras.layers.Conv2D(3, kernel_size, padding='same')(x)
                outputs = tf.keras.layers.Activation('sigmoid')(x)
                model = tf.keras.Model(inputs, outputs, name="DeepImagePrior")
                logger.info("Model created successfully")
                return model
            except Exception as e:
                logger.error(f"Error creating model: {str(e)}")
                raise ValueError(f"Error creating model: {str(e)}")

        # Optimized Deep Image Prior workflow
        def dip_workflow(x0, x_true, model, input_shape, num_iters=200, init_lr=0.01):
            try:
                logger.info("Starting DIP workflow")
                z = tf.random.uniform((1,) + input_shape, dtype=tf.float32)
                
                # Use a faster optimizer
                optimizer = tf.keras.optimizers.legacy.Adam(init_lr)
                model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
                
                # Reduce verbosity and use a simpler callback
                history = model.fit(
                    z, x_true[None, ...],
                    epochs=num_iters,
                    steps_per_epoch=1,
                    verbose=0,
                    callbacks=[TqdmCallback(verbose=0)]
                )

                x = model.predict(z, verbose=0)[0]
                logger.info("DIP workflow completed successfully")
                return x
            except Exception as e:
                logger.error(f"Error in workflow: {str(e)}")
                raise ValueError(f"Error in workflow: {str(e)}")

        try:
            # Load and process image with smaller size
            img = load_image(input_image_path)
            logger.info("Resizing image for processing")
            x_true = resize(img, (128, 128))  # Reduced size for faster processing
            x0 = get_noisy_img(x_true, sig)
            
            input_shape = x0.shape
            model = deep_image_prior(input_shape, noise_reg={'stddev': 1/30.})
            denoised_img = dip_workflow(x0, x_true, model, input_shape, num_iters, init_lr)

            # Resize back to original size
            logger.info("Resizing image back to original size")
            denoised_img = resize(denoised_img, img.shape[:2])
            
            # Save the denoised image
            logger.info(f"Saving denoised image to {output_image_path}")
            denoised_img = (denoised_img * 255).astype(np.uint8)
            plt.imsave(output_image_path, denoised_img)

            # Verify the file was saved
            if not os.path.exists(output_image_path):
                logger.error(f"Failed to save processed image to {output_image_path}")
                raise ValueError(f"Failed to save processed image to {output_image_path}")

            logger.info(f"Successfully created denoised image at {output_image_path}")
            return output_image_path

        except Exception as e:
            logger.error(f"Error during image processing: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    except Exception as e:
        logger.error(f"Unexpected error in denoise_image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

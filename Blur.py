import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def blur_image_cnn(input_image_path, kernel_size, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Image at path {input_image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create and normalize the kernel
    kernel = np.ones((kernel_size, kernel_size, 1, 1), np.float32) / (kernel_size ** 2)
    kernel = np.repeat(kernel, 3, axis=2)  # Repeat for 3 input channels (RGB)
    kernel = np.repeat(kernel, 3, axis=3)  # Repeat for 3 output channels (RGB)

    # Create and normalize the kernel
    # kernel = np.ones((kernel_size, kernel_size, 1, 1), np.float32) / (kernel_size ** 2)

    # Convert image to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Expand dimensions of the image to fit the model input requirements (batch size of 1)
    image = np.expand_dims(image, axis=0)

    # Create the CNN model with the custom kernel
    input_shape = (None, None, 3)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=3, kernel_size=kernel_size, padding='same', use_bias=False, 
                               kernel_initializer=tf.constant_initializer(kernel), input_shape=input_shape)
    ])

    # Compile the model to initialize weights
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Apply the model to the image
    blurred_image = model.predict(image)
    
    # Remove the batch dimension and convert back to [0, 255] range
    blurred_image = np.squeeze(blurred_image, axis=0)
    blurred_image = np.clip(blurred_image * 255, 0, 255).astype(np.uint8)

    # Convert the image back to BGR before saving
    blurred_image_bgr = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR)

    # Save the blurred image
    cv2.imwrite(output_image_path, blurred_image_bgr)
    
    return output_image_path

# Example usage
# input_image_path = '/content/drive/MyDrive/img_4.jpg'
# output_image_path = '/content/drive/MyDrive/blurred_img_4.jpg'
# kernel_size = 5

# blurred_image_path = blur_image_cnn(input_image_path, kernel_size, output_image_path)
# print(f"Blurred image saved at: {blurred_image_path}")
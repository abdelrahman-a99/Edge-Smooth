# import numpy as np
# from skimage.transform import resize
# from skimage.io import imread
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tqdm.keras import TqdmCallback

# def denoise_image(image_path, num_iters=100, init_lr=0.01, sig=30):
#     print(tf.version)
#     print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#     # Function to add noise to an image
#     def get_noisy_img(img, sig=30):
#         sigma = sig / 255.
#         noise = np.random.normal(scale=sigma, size=img.shape)
#         img_noisy = np.clip(img + noise, 0, 1).astype(np.float32)
#         return img_noisy

#     # Function to load an image from a file path
#     def load_image(image_path):
#         img = imread(image_path).astype(np.float32) / 255.
#         return img

#     # Function to display images
#     # def display_images(x0, x_true, x_denoised):
#     #     _, axis = plt.subplots(1, 3, figsize=(16, 5))
#     #     axis[0].imshow(x0)
#     #     axis[0].set_title("Noisy Image")
#     #     axis[1].imshow(x_true)
#     #     axis[1].set_title("Original Image")
#     #     axis[2].imshow(x_denoised)
#     #     axis[2].set_title("Denoised Image")
#     #     for ax in axis:
#     #         ax.set_axis_off()
#     #     plt.show()

#     # Deep Image Prior model definition
#     def deep_image_prior(input_shape, noise_reg=None, layers=(64, 128, 128, 64), kernel_size=3):
#         def norm_and_active(x):
#             x = tf.keras.layers.BatchNormalization()(x)
#             x = tf.keras.layers.LeakyReLU()(x)
#             return x

#         inputs = tf.keras.Input(shape=input_shape)
#         x = inputs
#         if noise_reg:
#             x = tf.keras.layers.GaussianNoise(noise_reg['stddev'])(x)

#         skips = []
#         for filters in layers:
#             x = norm_and_active(tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x))
#             skips.append(x)
#             x = norm_and_active(tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same')(x))

#         for filters in layers[::-1]:
#             x = tf.keras.layers.UpSampling2D()(x)
#             x = tf.keras.layers.Concatenate()([x, skips.pop()])
#             x = norm_and_active(tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x))

#         x = tf.keras.layers.Conv2D(3, kernel_size, padding='same')(x)
#         outputs = tf.keras.layers.Activation('sigmoid')(x)
#         return tf.keras.Model(inputs, outputs, name="DeepImagePrior")

#     # Deep Image Prior workflow
#     def dip_workflow(x0, x_true, model, input_shape, num_iters=1500, init_lr=0.01):
#         z = tf.random.uniform((1,) + input_shape, dtype=tf.float32)

#         model.compile(optimizer=tf.keras.optimizers.Adam(init_lr), loss=tf.keras.losses.MeanSquaredError())
#         history = model.fit(z, x_true[None, ...], epochs=num_iters, steps_per_epoch=1, verbose=0, callbacks=[TqdmCallback(verbose=1)])

#         x = model.predict(z)[0]
#         # display_images(x0, x_true, x)

#         # Edited from chat
#         return x

#     # Main function to run the Deep Image Prior model
#     img = load_image(image_path)
#     x_true = resize(img, (256, 256))
#     x0 = get_noisy_img(x_true, sig)

#     input_shape = x0.shape
#     model = deep_image_prior(input_shape, noise_reg={'stddev': 1/30.})
#     # dip_workflow(x0, x_true, model, input_shape, num_iters, init_lr)

#     # edited from chat
#     denoised_img = dip_workflow(x0, x_true, model, input_shape, num_iters, init_lr)

#     # return image_path

#     # edited from chat
#     # Save the denoised image
#     denoised_img = (denoised_img * 255).astype(np.uint8)
    
#     return denoised_img

# # Example usage
# # Replace 'your_image_path_here' with the actual path to your image
# # image_path = '/content/drive/MyDrive/img_4.jpg'
# # denoise_image(image_path)

import numpy as np
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.keras import TqdmCallback
import os

def denoise_image(image_path, output_path, num_iters=1000, init_lr=0.01, sig=30):
    print(tf.version)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Function to add noise to an image
    def get_noisy_img(img, sig=30):
        sigma = sig / 255.
        noise = np.random.normal(scale=sigma, size=img.shape)
        img_noisy = np.clip(img + noise, 0, 1).astype(np.float32)
        return img_noisy

    # Function to load an image from a file path
    def load_image(image_path):
        img = imread(image_path).astype(np.float32) / 255.
        # return img
        # Ensure image is RGB
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img

    # Deep Image Prior model definition
    def deep_image_prior(input_shape, noise_reg=None, layers=(64, 128, 128, 64), kernel_size=3):
        def norm_and_active(x):
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            return x

        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        if noise_reg:
            x = tf.keras.layers.GaussianNoise(noise_reg['stddev'])(x)

        skips = []
        for filters in layers:
            x = norm_and_active(tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x))
            skips.append(x)
            x = norm_and_active(tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same')(x))

        for filters in layers[::-1]:
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = norm_and_active(tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x))

        x = tf.keras.layers.Conv2D(3, kernel_size, padding='same')(x)
        outputs = tf.keras.layers.Activation('sigmoid')(x)
        return tf.keras.Model(inputs, outputs, name="DeepImagePrior")

    # Deep Image Prior workflow
    def dip_workflow(x0, x_true, model, input_shape, num_iters=1000, init_lr=0.01):
        z = tf.random.uniform((1,) + input_shape, dtype=tf.float32)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(init_lr), loss=tf.keras.losses.MeanSquaredError())
        history = model.fit(z, x_true[None, ...], epochs=num_iters, steps_per_epoch=1, verbose=0, callbacks=[TqdmCallback(verbose=1)])

        x = model.predict(z)[0]
        return x

    # Main function to run the Deep Image Prior model
    img = load_image(image_path)
    x_true = resize(img, (256, 256))
    x0 = get_noisy_img(x_true, sig)
    
    input_shape = x0.shape
    model = deep_image_prior(input_shape, noise_reg={'stddev': 1/30.})
    denoised_img = dip_workflow(x0, x_true, model, input_shape, num_iters, init_lr)

    # Save the denoised image
    denoised_img = (denoised_img * 255).astype(np.uint8)
    plt.imsave(output_path, denoised_img)

    return output_path
    # return denoised_img

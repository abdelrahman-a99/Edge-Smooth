import numpy as np
import cv2

# import matplotlib.pyplot as plt
from PIL import Image


def Edge_Detection(path):
    # print("Loading image from path:", path)

    Fence_Gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if Fence_Gray is None:
        print("Error: Unable to load the image from the specified path.")
        return None

    testImageHeight = Fence_Gray.shape[0]
    testImageWidth = Fence_Gray.shape[1]

    # print("continue after shape")

    # imageNChannels = 1
    # Sobel operator for horizontal edge detection
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Sobel operator for vertical edge detection
    Ky = Kx.T

    # Without using padding, the output array is 2 pixels shorter on each dimension
    convolvedShape = (testImageHeight - 2, testImageWidth - 2)
    Fence_Gray_Edges1x = np.zeros(convolvedShape)  # Horiz
    Fence_Gray_Edges1y = np.zeros(convolvedShape)  # Vertical
    Fence_Gray_Edges1 = np.zeros(convolvedShape)  # Combined
    for i in range(1, testImageHeight - 1):
        for j in range(1, testImageWidth - 1):
            window = Fence_Gray[i - 1 : i + 2, j - 1 : j + 2]
            a = Fence_Gray_Edges1x[i - 1, j - 1] = np.maximum(
                np.sum(np.multiply(window, Kx)), 0
            )
            b = Fence_Gray_Edges1y[i - 1, j - 1] = np.maximum(
                np.sum(np.multiply(window, Ky)), 0
            )
            Fence_Gray_Edges1[i - 1, j - 1] = np.sqrt(a**2 + b**2)

    # Fence_Gray_Edges1 = cv2.normalize(Fence_Gray_Edges1, None, 0, 255, cv2.NORM_MINMAX)
    # Fence_Gray_Edges1 = np.uint8(Fence_Gray_Edges1)
    final_image = Image.fromarray(Fence_Gray_Edges1.astype(np.uint8))

    # final_image.save("C:\\Users\\WIN10\\Desktop\\Desktop\\test\\static\\processed\\processed_image.png")
    return final_image

# path = 'black-metal-fence-and-floor-tile-beside-the-river-at-night.jpg'
# path = "Screenshot 2024-05-12 232722.png"
# Edge_Detection(path)

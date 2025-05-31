import cv2

def negative_transform_color(input_path, output_path):
    img_bgr = cv2.imread(input_path, 1)

    height, width, _ = img_bgr.shape

    for i in range(height):
        for j in range(width):
            pixel = img_bgr[i, j]
            pixel[0] = 255 - pixel[0]  # Invert Blue
            pixel[1] = 255 - pixel[1]  # Invert Green
            pixel[2] = 255 - pixel[2]  # Invert Red
            img_bgr[i, j] = pixel

    cv2.imwrite(output_path, img_bgr)

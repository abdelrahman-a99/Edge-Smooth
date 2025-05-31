# Image Processing Web Application

This is a web application that provides various image processing capabilities including:
- Edge Detection
- Image Smoothing
- Blur Effects
- Negative Transformation

## Features

- **Edge Detection**: Implements Sobel edge detection algorithm
- **Image Smoothing**: Uses Deep Image Prior for image denoising
- **Blur Effects**: CNN-based image blurring
- **Negative Transformation**: Color and grayscale image inversion

## Requirements

- Python 3.8+
- Flask
- OpenCV
- TensorFlow
- NumPy
- scikit-image
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abdelrahman-a99/Edge-Smooth.git
cd Edge-Smooth
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image and select the desired processing algorithm.

## Project Structure

```
├── app.py                 # Main Flask application
├── Blur.py               # Blur effect implementation
├── Edge_detection.py     # Edge detection implementation
├── Negative_Transformation.py  # Negative transformation implementation
├── Smoothing.py          # Image smoothing implementation
├── static/              # Static files (CSS, JS, images)
│   ├── uploads/        # Uploaded images
│   └── processed/      # Processed images
└── templates/          # HTML templates
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License

Copyright (c) 2024 Abdelrahman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
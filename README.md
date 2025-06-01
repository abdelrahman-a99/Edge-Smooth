# Edge Smooth - Image Processing Application

A Flask-based web application for advanced image processing, featuring edge detection, smoothing, and other image enhancement techniques.

## Features

- **Edge Detection**: Identify and highlight edges in images using advanced algorithms
- **Image Smoothing**: Apply deep learning-based smoothing to reduce noise while preserving important details
- **Negative Transformation**: Convert images to their negative form in both grayscale and color modes
- **CNN-based Blur**: Apply convolutional neural network-based blur effects
- **Real-time Preview**: See the original and processed images side by side
- **Progress Tracking**: Visual feedback during image processing
- **Error Handling**: Comprehensive error handling and user feedback

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, TensorFlow, scikit-image
- **Deep Learning**: TensorFlow for image smoothing
- **Development**: Git for version control

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/abdelrahman-a99/Edge-Smooth.git
   cd Edge-Smooth
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app/app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage

1. **Upload an Image**:
   - Click the upload button or drag and drop an image
   - Supported formats: JPG, JPEG, PNG
   - Maximum file size: 16MB

2. **Select Processing Algorithm**:
   - Edge Detection: Highlights edges in the image
   - Smoothing: Reduces noise while preserving details
   - Negative Transformation: Converts image to negative form
   - CNN Blur: Applies neural network-based blur

3. **Process Image**:
   - Click the "Process" button
   - Wait for the processing to complete
   - View the results in the preview panel

4. **Reset**:
   - Use the reset button to start over with a new image

## Project Structure

```
Edge-Smooth/
├── app/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── uploads/
│   │   └── processed/
│   ├── templates/
│   └── utils/
│       ├── Edge_detection.py
│       ├── Smoothing.py
│       ├── Negative_Transformation.py
│       └── Blur.py
├── requirements.txt
└── README.md
```

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- File size limits
- Processing errors
- Server-side issues

All errors are logged and displayed to the user with clear messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Flask team for the web framework
- OpenCV and scikit-image teams for image processing libraries 
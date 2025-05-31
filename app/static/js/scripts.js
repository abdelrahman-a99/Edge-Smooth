document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file');
    const submitBtn = document.getElementById('submitBtn');
    const resetBtn = document.getElementById('resetBtn');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const spinner = document.getElementById('spinner');
    const messageContainer = document.getElementById('messageContainer');
    const originalPreview = document.getElementById('originalPreview');
    const processedPreview = document.getElementById('processedPreview');
    const algorithmSelect = document.getElementById('algorithm');

    // Initially disable algorithm selection and buttons
    algorithmSelect.disabled = true;
    submitBtn.disabled = true;
    resetBtn.disabled = true;

    // File size limit (16MB)
    const MAX_FILE_SIZE = 16 * 1024 * 1024;
    // Allowed file types
    const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
    // Processing timeout (30 seconds)
    const PROCESSING_TIMEOUT = 30000;

    // Show message function
    function showMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.textContent = message;
        messageContainer.appendChild(messageDiv);
        messageDiv.style.display = 'block';

        // Remove message after 5 seconds
        setTimeout(() => {
            messageDiv.style.display = 'none';
            messageDiv.remove();
        }, 5000);
    }

    // Function to disable all controls during processing
    function disableControls() {
        algorithmSelect.disabled = true;
        submitBtn.disabled = true;
        resetBtn.disabled = true;
        fileInput.disabled = true;
    }

    // Function to enable controls after processing
    function enableControls() {
        algorithmSelect.disabled = false;
        submitBtn.disabled = false;
        fileInput.disabled = false;
        // Reset button state depends on whether there's a processed image
        resetBtn.disabled = !processedPreview.src;
    }

    // Reset function
    function resetForm() {
        if (!fileInput.files[0]) {
            resetBtn.disabled = true;
            return;
        }
        
        algorithmSelect.disabled = false;
        algorithmSelect.value = '';
        submitBtn.disabled = true;
        processedPreview.src = '';
        progressContainer.style.display = 'none';
        spinner.style.display = 'none';
        progressBar.style.width = '0%';
        progressText.textContent = 'Processing: 0%';
        resetBtn.disabled = true;
    }

    // Reset button click handler
    resetBtn.addEventListener('click', resetForm);

    // File validation
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        
        if (file) {
            // Check file size
            if (file.size > MAX_FILE_SIZE) {
                showMessage('File size exceeds 16MB limit', 'error');
                fileInput.value = '';
                disableControls();
                processedPreview.src = '';
                return;
            }

            // Check file type
            if (!ALLOWED_TYPES.includes(file.type)) {
                showMessage('Invalid file type. Please upload JPG, PNG, or JPEG', 'error');
                fileInput.value = '';
                disableControls();
                processedPreview.src = '';
                return;
            }

            // Preview original image
            const reader = new FileReader();
            reader.onload = function(e) {
                originalPreview.src = e.target.result;
            };
            reader.readAsDataURL(file);

            // Enable algorithm selection when valid image is uploaded
            algorithmSelect.disabled = false;
            algorithmSelect.value = '';
            submitBtn.disabled = true;
            resetBtn.disabled = true;
            processedPreview.src = '';
        } else {
            // If no file is selected, disable everything
            disableControls();
            originalPreview.src = '';
            processedPreview.src = '';
        }
    });

    // Algorithm selection handler
    algorithmSelect.addEventListener('change', function() {
        if (this.value) {
            submitBtn.disabled = false;
            resetBtn.disabled = true; // Keep reset disabled until processing is complete
        } else {
            submitBtn.disabled = true;
            resetBtn.disabled = true;
        }
    });

    // Form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const file = fileInput.files[0];
        const algorithm = algorithmSelect.value;

        if (!file) {
            showMessage('Please select an image', 'error');
            return;
        }

        if (!algorithm) {
            showMessage('Please select an algorithm', 'error');
            return;
        }

        // Disable all controls during processing
        disableControls();

        // Show progress and spinner
        progressContainer.style.display = 'block';
        spinner.style.display = 'block';

        // Create FormData with the original file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('algorithm', algorithm);

        // Simulate progress with slower, more realistic animation
        let progress = 0;
        const progressInterval = setInterval(() => {
            // Use a very slow, non-linear progress increment
            const remaining = 90 - progress;
            const increment = Math.max(0.05, remaining * 0.01); // Much slower increment
            progress = Math.min(90, progress + increment);
            
            progressBar.style.width = `${progress.toFixed(1)}%`;
            progressText.textContent = `Processing: ${Math.round(progress)}%`;
        }, 100); // Update every 100ms for smoother animation

        // Set a timeout for the processing
        const processingTimeout = setTimeout(() => {
            clearInterval(progressInterval);
            showMessage('Processing is taking longer than expected. Please wait...', 'error');
        }, PROCESSING_TIMEOUT);

        // Send request
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Processing failed');
                });
            }
            return response.json();
        })
        .then(data => {
            clearInterval(progressInterval);
            clearTimeout(processingTimeout);
            progressBar.style.width = '100%';
            progressText.textContent = 'Processing: 100%';
            
            if (data.success) {
                // Show processed image with timestamp to prevent caching
                const timestamp = new Date().getTime();
                processedPreview.src = `/static/processed/${data.filename}?t=${timestamp}`;
                showMessage('Image processed successfully!', 'success');
                
                // Enable reset button only after successful processing
                resetBtn.disabled = false;
            } else {
                throw new Error(data.error || 'Processing failed');
            }
        })
        .catch(error => {
            clearInterval(progressInterval);
            clearTimeout(processingTimeout);
            showMessage('Error processing image: ' + error.message, 'error');
            resetBtn.disabled = true;
        })
        .finally(() => {
            // Hide progress and spinner after a delay
            setTimeout(() => {
                progressContainer.style.display = 'none';
                spinner.style.display = 'none';
                enableControls();
                progressBar.style.width = '0%';
                progressText.textContent = 'Processing: 0%';
            }, 1000);
        });
    });
});

document.querySelector('form').onsubmit = function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const img = document.getElementById('processedImage');
        img.src = url;
        img.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
    });
};

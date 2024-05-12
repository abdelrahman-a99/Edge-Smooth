function previewImage(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function() {
        const img = document.createElement('img');
        img.src = reader.result;
        document.getElementById('imagePreview').innerHTML = '';
        document.getElementById('imagePreview').appendChild(img);
    };

    if (file) {
        reader.readAsDataURL(file);
    }
}

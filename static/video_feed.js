const functionSelect = document.getElementById('function');
const videoElement = document.getElementById('video');

functionSelect.addEventListener('change', () => {
    const selectedFunction = functionSelect.value;
    videoElement.src = `/video_feed?function=${selectedFunction}`;
});

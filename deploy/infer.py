from flask import Flask, request, jsonify, Response
import onnxruntime
import numpy as np
from PIL import Image
import io
import sys
import base64

# --- Configuration ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MNIST Test Page</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
        #canvas { border: 2px solid #555; background: black; cursor: crosshair; touch-action: none; }
        .controls { margin-top: 20px; }
        .result { margin-top: 30px; font-size: 1.5em; }
        .prediction-text { font-weight: bold; color: #008000; }
        button { padding: 10px 20px; font-size: 16px; margin: 0 5px; cursor: pointer; }
    </style>
</head>
<body>

    <h1>Draw a Digit (0-9)</h1>
    <p>Use a thick white line to draw the digit inside the box.</p>

    <canvas id="canvas" width="300" height="300"></canvas>

    <div class="controls">
        <button onclick="predictDigit()">Predict</button>
        <button onclick="clearCanvas()">Clear</button>
    </div>

    <div class="result">
        Predicted Digit: <span id="prediction" class="prediction-text">?</span>
        <br>
        Score: <span id="score">N/A</span>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        // --- Setup Context ---
        ctx.lineWidth = 20;        // Thickness of the line
        ctx.lineCap = 'round';     // Rounded ends
        ctx.strokeStyle = 'white'; // White digit

        // --- Drawing Handlers ---
        function getMousePos(e) {
            const rect = canvas.getBoundingClientRect();
            // Handle touch events by using the first touch point
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: clientX - rect.left,
                y: clientY - rect.top
            };
        }

        function startDrawing(e) {
            e.preventDefault(); // Prevent default touch behavior (e.g., scrolling)
            isDrawing = true;
            const pos = getMousePos(e);
            [lastX, lastY] = [pos.x, pos.y];
        }

        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }

        function draw(e) {
            if (!isDrawing) return;
            e.preventDefault(); // Prevent default touch behavior (e.g., scrolling)

            const pos = getMousePos(e);

            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();

            [lastX, lastY] = [pos.x, pos.y];
        }

        // --- Event Listeners ---
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        canvas.addEventListener('mousemove', draw);

        // Touch event listeners
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchend', stopDrawing);
        canvas.addEventListener('touchcancel', stopDrawing);
        canvas.addEventListener('touchmove', draw);


        // --- Control Functions ---
        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('prediction').textContent = '?';
            document.getElementById('score').textContent = 'N/A';
        }

        // Initialize canvas to black background when script loads
        window.onload = clearCanvas;


        function predictDigit() {
            // 1. Capture the canvas content as a Base64 PNG data URI
            const imageDataURI = canvas.toDataURL('image/png');

            // 2. Send the data to the Flask /predict endpoint
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageDataURI }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('prediction').textContent = 'Error';
                    document.getElementById('score').textContent = data.error;
                    document.getElementById('prediction').style.color = 'red';
                } else if (data.message) {
                    document.getElementById('prediction').textContent = 'N/A';
                    document.getElementById('score').textContent = data.message;
                    document.getElementById('prediction').style.color = '#008000';
                } else {
                    document.getElementById('prediction').textContent = data.prediction;
                    document.getElementById('score').textContent = data.score;
                    document.getElementById('prediction').style.color = '#008000';
                }
            })
            .catch(error => {
                console.error('Fetch Error:', error);
                document.getElementById('prediction').textContent = 'Server Error';
                document.getElementById('score').textContent = 'Check console';
                document.getElementById('prediction').style.color = 'red';
            });
        }
    </script>
</body>
</html>
"""

ONNX_MODEL_PATH = "model.onnx"
PIXEL_SIZE = 28  # Model input size (1x28x28)

app = Flask(__name__)

# Load Model
try:
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX Model loaded. Input: {input_name}, Output: {output_name}")
except Exception as e:
    session = None
    print(f"FATAL ERROR: Could not load model '{ONNX_MODEL_PATH}': {e}")

# --- Preprocessing Function ---
def preprocess_image_data(image_data_uri):
    """
    Takes the Base64 image data URI from the front-end, decodes it,
    resizes, and prepares it for ONNX inference in [1, 1, 28, 28] format.
    """
    # Decode Base64 Image Data URI
    try:
        # Split data URI: data:image/png;base64,...
        _, encoded = image_data_uri.split(",", 1)
        image_bytes = io.BytesIO(base64.b64decode(encoded))
        img = Image.open(image_bytes).convert('L') # Convert to Grayscale
    except Exception as e:
        print(f"Image decoding failed: {e}")
        return None

    # Resize and Convert
    resized_img = img.resize((PIXEL_SIZE, PIXEL_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(resized_img, dtype=np.float32)

    # Normalize and Reshape
    # Assuming front-end draws white digit (1.0) on black background (0.0)
    input_tensor = img_array / 255.0

    # Reshape for ONNX: [Batch_Size, Channel, Height, Width]
    input_tensor = input_tensor.reshape(1, 1, PIXEL_SIZE, PIXEL_SIZE).astype(np.float32)

    # Check if the canvas is blank
    if np.all(input_tensor < 0.01):
        return None

    return input_tensor

# --- Flask Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the JavaScript front-end."""
    if session is None:
        return jsonify({'error': 'Model not initialized on server.'}), 500

    data = request.get_json()
    image_data_uri = data.get('image')

    input_tensor = preprocess_image_data(image_data_uri)

    if input_tensor is None:
        return jsonify({'prediction': 'N/A', 'score': 'N/A', 'message': 'Please draw a digit.'})

    # Run ONNX Inference
    try:
        raw_output = session.run([output_name], {input_name: input_tensor})

        # Postprocess: Convert logits to probabilities and find prediction
        raw_output = np.array(raw_output).flatten()

        print("Mnist output:", raw_output, file=sys.stderr)

        # Use softmax
        #if np.sum(raw_output) - 1.0 > 1e-3:
        #    logits = raw_output
        #    exp_logits = np.exp(logits - np.max(logits))
        #    probabilities = exp_logits / np.sum(exp_logits)
        #else:
        #    probabilities = raw_output
        #prediction = int(np.argmax(probabilities))
        #score = float(probabilities[prediction])

        scores = raw_output
        prediction = int(np.argmax(scores))
        score = float(scores[prediction])

        print(f"Prediction: item:{prediction} score:{score}", file=sys.stderr)

        return jsonify({
            'prediction': prediction,
            'score': score
        })

    except Exception as e:
        return jsonify({'error': f'Inference failed: {e}'}), 500

@app.route('/')
def index():
    """Renders the entire HTML/JS front-end."""

    # This entire block is the HTML/JavaScript for the front-end
    return Response(html_content, mimetype='text/html')

if __name__ == '__main__':
    # Set host='0.0.0.0' for external (outside the docker image) network access.
    print("\nStarting Flask server. Access the app at http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

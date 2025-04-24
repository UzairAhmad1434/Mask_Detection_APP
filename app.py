from flask import Flask, request, render_template, flash, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = str(uuid.uuid4())
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists and has correct permissions
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.chmod(app.config['UPLOAD_FOLDER'], 0o755)  # Ensure the folder is readable/writable

# Define the model path
MODEL_PATH = 'model.h5'  # Update if model is elsewhere, e.g., '/path/to/model.h5'

# Load the pre-trained face mask detection model
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the model file exists.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        # Load and preprocess the image for the model
        img = load_img(image_path, target_size=(224, 224))  # Adjust size based on your model
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize if your model expects this
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def predict_mask(image_path):
    try:
        # Preprocess image and predict
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)
        # Adjust based on your model's output (e.g., sigmoid or softmax)
        label = 'With Mask' if prediction[0][0] > 0.5 else 'Without Mask'
        confidence = prediction[0][0] if label == 'With Mask' else 1 - prediction[0][0]
        return label, confidence
    except Exception as e:
        raise ValueError(f"Error predicting mask: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Verify the file was saved
            if not os.path.exists(file_path):
                flash(f'Failed to save the image to {file_path}', 'error')
                return redirect(request.url)
            try:
                # Predict mask
                label, confidence = predict_mask(file_path)
                # Construct the relative path for the static folder
                relative_path = os.path.join('uploads', filename).replace('\\', '/')  # Normalize path for URLs
                image_url = url_for('static', filename=relative_path)
                print(f"Image saved at: {file_path}")
                print(f"Relative path: {relative_path}")
                print(f"Image URL: {image_url}")
                return render_template('result.html', 
                                     image_path=relative_path, 
                                     label=label, 
                                     confidence=f"{confidence:.2%}")
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file format. Please upload PNG, JPG, or JPEG.', 'error')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
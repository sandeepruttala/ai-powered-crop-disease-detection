import os
from PIL import Image
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from groq import Groq
from flask import jsonify
from flask_cors import CORS
import markdown
from pymongo import MongoClient
import gridfs
from bson import ObjectId
from datetime import datetime

MONGO_URI = "mongodb+srv://dev-user-1:pass-word-dev@cluster0.hdz2b.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["crop_disease_db"]
fs = gridfs.GridFS(db)
collection = db["images"]


app = Flask(__name__)
CORS(app)

# Set up working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/models/ai_crop_final.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# List of classes
classes = ['Apple___Apple_scab',
           'Apple___Black_rot',
           'Apple___Cedar_apple_rust',
           'Apple___healthy',
           'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew',
           'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
           'Corn_(maize)___Common_rust_',
           'Corn_(maize)___Northern_Leaf_Blight',
           'Corn_(maize)___healthy',
           'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)',
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
           'Grape___healthy',
           'Orange___Haunglongbing_(Citrus_greening)',
           'Peach___Bacterial_spot',
           'Peach___healthy',
           'Pepper,_bell___Bacterial_spot',
           'Pepper,_bell___healthy',
           'Potato___Early_blight',
           'Potato___Late_blight',
           'Potato___healthy',
           'Raspberry___healthy',
           'Soybean___healthy',
           'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch',
           'Strawberry___healthy',
           'Tomato___Bacterial_spot',
           'Tomato___Early_blight',
           'Tomato___Late_blight',
           'Tomato___Leaf_Mold',
           'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite',
           'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
           'Tomato___Tomato_mosaic_virus',
           'Tomato___healthy']

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = classes[predicted_class_index]
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Received POST request")
        if 'file' not in request.files:
            print("No file part in request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            print(f"Saving file to {file_path}")
            file.save(file_path)
            
            # Get the prediction label
            label = predict_image_class(model, file_path)

            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Store in MongoDB
            prediction_data = {
                "filename": file.filename,
                "label": label,
                "timestamp": timestamp
            }
            collection.insert_one(prediction_data)  # Save to MongoDB
            
            return render_template('app.html', label=label, file_path=file.filename)
    return render_template('app.html')

@app.route('/label', methods=['GET', 'POST'])
def upload_file_mobile():
    if request.method == 'POST':
        print("Received POST request")
        if 'file' not in request.files:
            print("No file part in request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            print(f"Saving file to {file_path}")
            file.save(file_path)
            label = predict_image_class(model, file_path)
            return jsonify({'label': label})
    return jsonify({'error': 'No file uploaded'})


@app.route('/remidies', methods=['GET', 'POST'])
def remidies():
    if request.method == 'POST':
        data = request.get_json()
        disease_name = data.get('disease_name')

        if not disease_name:
            return jsonify({'error': 'Disease name is required'})
        
        content = f"Generate remidies for {disease_name}"
        
        client = Groq(
            api_key=os.environ.get("CROP_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="llama-3.3-70b-versatile",
        )
        
        raw = chat_completion.choices[0].message.content
        processed = markdown.markdown(raw)
        response = {'remidies': processed}
        return jsonify(response)
    
@app.route('/history')
def history():
    # Fetch the last 10 predictions (latest first)
    predictions = list(collection.find().sort("timestamp", -1).limit(10))

    return render_template('history.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
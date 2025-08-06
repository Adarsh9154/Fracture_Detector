from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime
from flask import send_file, request
from io import BytesIO
from reportlab.pdfgen import canvas
from flask import session
from reportlab.lib.utils import ImageReader

app = Flask(__name__)

model = load_model("best_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    upload_dir = os.path.join('static', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    prediction = model.predict(input_data)[0][0]
    result = "Normal" if prediction > 0.5 else "Fractured"
    confidence = round((prediction if result == "Normal" else 1 - prediction) * 100, 2)

    doctor_note = "Please consult your physician if symptoms persist."
    current_date = datetime.now().strftime("%B %d, %Y")

    return render_template('result.html', prediction=result, accuracy=confidence, image_path=filepath,doctor_note=doctor_note,
                       current_date=current_date)


@app.route('/download_report')
def download_report():
    prediction = request.args.get("prediction", "Unknown")
    accuracy = request.args.get("accuracy", "N/A")
    image_path = request.args.get("image_path", None)
    current_date = request.args.get("current_date", "")
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 770, "X-ray Analysis Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 740, f"Date: {current_date}")
    c.drawString(100, 720, f"Diagnosis: {prediction}")
    c.drawString(100, 700, f"Accuracy: {accuracy}%")

    if image_path and os.path.exists(image_path):
        c.drawImage(ImageReader(image_path), 100, 400, width=300, height=300)

    c.drawString(100, 380, "Doctor's Note:")
    c.setFont("Helvetica-Oblique", 12)
    c.drawString(120, 360, "Always consult a medical professional.")

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="xray_report.pdf", mimetype="application/pdf")

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, send_file
from ultralytics import YOLO
import os, uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ✅ Load only best.pt model once
print("Loading YOLO model... please wait ⏳")
model = YOLO("weights/best.pt")
print("✅ Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save uploaded file
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run YOLO prediction
    results = model.predict(
        source=filepath,
        save=True,
        project=app.config['RESULT_FOLDER'],
        name='',
        exist_ok=True
    )

    # Find saved result file
    result_dir = results[0].save_dir
    result_name = os.path.basename(filepath)
    result_path = os.path.join(result_dir, result_name)
    if not os.path.exists(result_path):
        for f in os.listdir(result_dir):
            if os.path.splitext(f)[0] in os.path.splitext(result_name)[0]:
                result_path = os.path.join(result_dir, f)
                break

    # Classes detected
    detected_classes = [model.names[int(c)] for c in results[0].boxes.cls] if results[0].boxes else []
    result_url = result_path.replace("\\", "/")

    return render_template(
        'result.html',
        result_file=result_url,
        detected_classes=list(set(detected_classes))
    )

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,debug=True)


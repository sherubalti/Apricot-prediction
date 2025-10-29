from flask import Flask, render_template, request, redirect, send_file
from ultralytics import YOLO
import os, uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load models once (Replit can be slow, so just load when accessed)
model_best, model_last = None, None

def load_model(model_name):
    global model_best, model_last
    if model_name == 'best':
        if model_best is None:
            model_best = YOLO("weights/best.pt")
        return model_best
    else:
        if model_last is None:
            model_last = YOLO("weights/last.pt")
        return model_last

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

    model_choice = request.form.get('model_choice')
    model = load_model(model_choice)

    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    results = model.predict(
        source=filepath,
        save=True,
        project=app.config['RESULT_FOLDER'],
        name='',
        exist_ok=True
    )

    result_dir = results[0].save_dir
    result_name = os.path.basename(filepath)
    result_path = os.path.join(result_dir, result_name)
    if not os.path.exists(result_path):
        for f in os.listdir(result_dir):
            if os.path.splitext(f)[0] in os.path.splitext(result_name)[0]:
                result_path = os.path.join(result_dir, f)
                break

    result_url = result_path.replace("\\", "/")
    detected_classes = [model.names[int(c)] for c in results[0].boxes.cls] if results[0].boxes else []

    return render_template(
        'result.html',
        result_file=result_url,
        detected_classes=list(set(detected_classes)),
        model_used=model_choice
    )

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

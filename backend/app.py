from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
from predict import predict_animal  # your prediction function

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')  # inside backend/static/uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    uploaded_image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predictions = predict_animal(filepath)
            uploaded_image_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', predictions=predictions, uploaded_image_url=uploaded_image_url)

if __name__ == '__main__':
    app.run(debug=True)

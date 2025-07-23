from flask import Flask, render_template_string, request
import os
import tempfile

from accentrecog.predict import predict_file

MODEL_PATH = os.getenv("MODEL_PATH", "accent_model.pt")

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Accent Recognition</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Roboto', sans-serif; max-width: 600px; margin: 40px auto; text-align: center; }
        h1 { font-weight: 500; }
        form { margin-top: 20px; }
        input[type=file], input[type=text], button { width: 100%; padding: 10px; margin: 10px 0; }
        .result { margin-top: 20px; font-size: 1.2em; }
    </style>
</head>
<body>
    <h1>Accent Recognition</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="audio" accept=".wav" required>
        <input type="text" name="model" placeholder="Model path" value="{{model_path}}" />
        <button type="submit">Predict</button>
    </form>
    {% if result %}
    <div class="result">Prediction: <strong>{{ result }}</strong></div>
    {% endif %}
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    model_path = MODEL_PATH
    if request.method == 'POST':
        audio = request.files.get('audio')
        model_path = request.form.get('model', MODEL_PATH)
        if audio:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                audio.save(tmp.name)
                result = predict_file(model_path, tmp.name)
            os.unlink(tmp.name)
    return render_template_string(INDEX_HTML, result=result, model_path=model_path)


if __name__ == '__main__':
    app.run(debug=True)

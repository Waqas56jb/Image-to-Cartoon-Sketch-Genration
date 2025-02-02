import os
import cv2
import numpy as np
import uuid
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# Ensure output folder exists
OUTPUT_FOLDER = "static/output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def increase_brightness(image, gamma=1.8):
    """Adjusts image brightness using gamma correction."""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_colors(image):
    """Boosts color vibrancy to make the image look more cartoon-like."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def create_cartoon_avatar(image_path):
    """Creates a highly polished cartoon avatar with exaggerated artistic effects."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    smooth = cv2.bilateralFilter(img, d=25, sigmaColor=300, sigmaSpace=300)
    smooth = increase_brightness(smooth, gamma=1.9)
    smooth = enhance_colors(smooth)
    smooth_blur = cv2.GaussianBlur(smooth, (15, 15), 0)
    cartoon = cv2.addWeighted(smooth, 0.8, smooth_blur, 0.2, 0)
    cartoon = cv2.bitwise_and(cartoon, cartoon, mask=edges)

    unique_filename = f"cartoon_{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, unique_filename)
    cv2.imwrite(output_path, cartoon)
    
    return unique_filename

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    image_path = os.path.join(OUTPUT_FOLDER, file.filename)
    file.save(image_path)
    cartoon_filename = create_cartoon_avatar(image_path)

    if cartoon_filename:
        return {"cartoon_image": cartoon_filename}
    else:
        return "Processing failed", 500

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

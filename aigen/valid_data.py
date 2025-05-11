from flask import Flask, send_file, jsonify, request, Response
import os
import json

html_content = """
<!DOCTYPE html>
<html lang="fr">
  <head>
    <meta charset="UTF-8" />
    <title>Validation Dataset</title>
    <style>
      body {
        font-family: sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        background: linear-gradient(to right, #f44336 0%, #4caf50 100%);
        min-height: 100vh;
        margin: 0;
        padding: 20px;
      }
      #card {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: white;
        padding: 20px;
        border-radius: 20px;
        margin-top: 30px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
      }
      .image-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 40px;
        margin-top: 10px;
      }
      .image-container {
        text-align: center;
      }
      .caption {
        font-weight: bold;
      }
      img {
        width: 400px;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
      }
      .buttons {
        margin-top: 30px;
        display: flex;
        gap: 60px;
      }
      button {
        padding: 20px 30px;
        border: none;
        border-radius: 10px;
        font-size: 20px;
        cursor: pointer;
      }
      .left {
        background: #f44336;
        color: white;
      }
      .right {
        background: #4caf50;
        color: white;
      }
      #info {
        margin-top: 10px;
        font-size: 18px;
        color: #333;
        font-weight: bold;
      }
      #filename {
        font-weight: bold;
        font-size: 24px;
        margin: 10px 0;
      }
      #caption-text {
        font-style: italic;
        font-size: 16px;
        color: #555;
        text-align: center;
        margin-bottom: 10px;
      }
      #deleted {
        margin-top: 20px;
        font-size: 16px;
        font-weight: bold;
        color: #222;
      }
    </style>
  </head>
  <body>
    <h1>🚀 Dataset Validation 🚀</h1>
    <div id="info"></div>
    <div id="card"></div>
    <div class="buttons">
      <button class="left" onclick="reject()">❌ Reject</button>
      <button class="right" onclick="accept()">✅ Keep</button>
    </div>
    <div id="deleted"></div>

    <script>
      let photos = [];
      let currentIndex = 0;
      let deletedCount = 0;

      fetch("/list")
        .then((res) => res.json())
        .then((data) => {
          console.log("Fetched images :", data);
          photos = data;
          showNextPair();
        })
        .catch((err) => {
          console.error("Failed to fetch /list :", err);
        });

      function showNextPair() {
        const container = document.getElementById("card");
        const info = document.getElementById("info");
        const deleted = document.getElementById("deleted");
        container.innerHTML = "";

        if (currentIndex >= photos.length) {
          container.innerHTML = "<h2>✅ Dataset Cleaned !</h2>";
          info.innerText = "";
          deleted.innerText = `Deleted Counter : ${deletedCount}`;
          return;
        }

        const pair = photos[currentIndex];

        const realURL = `/file/${encodeURIComponent(pair.real)}`;
        const genURL = `/file/${encodeURIComponent(pair.generated)}`;
        const cleanName = pair.real.split(/[\\/]/).pop().replace(".png", "");

        container.innerHTML = `
          <div id="filename">${cleanName}</div>
          <div id="caption-text">${pair.caption || ""}</div>
          <div class="image-row">
            <div class="image-container">
              <div class="caption">AI Version</div>
              <img src="${genURL}" alt="Generated Image">
            </div>
            <div class="image-container">
              <div class="caption">Real Version</div>
              <img src="${realURL}" alt="Real Image">
            </div>
          </div>
        `;

        info.innerText = `Image ${currentIndex + 1} / ${photos.length}`;
        deleted.innerText = `Deleted Counter : ${deletedCount}`;
      }

      function accept() {
        console.log("Kept :", photos[currentIndex].real);
        currentIndex++;
        showNextPair();
      }

      function reject() {
        const pair = photos[currentIndex];
        fetch("/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(pair),
        }).then(() => {
          console.log("Deleted :", pair.real);
          deletedCount++;
          currentIndex++;
          showNextPair();
        });
      }

      // Keyboard shortcuts
      document.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight") accept();
        if (e.key === "ArrowLeft") reject();
      });
    </script>
  </body>
</html>
"""

app = Flask(__name__, template_folder='')
BASE_PATH = os.path.abspath(".")
INPUT_PATH = os.path.join(BASE_PATH, "aigen", "inputs")
CAPTIONS_PATH = os.path.join(INPUT_PATH, "captions.json")


@app.route("/")
def index():
    return Response(html_content, mimetype="text/html")


@app.route("/list")
def list_images():
    files = os.listdir(INPUT_PATH)
    real_photos = [f for f in files if f.endswith(
        '.png') and os.path.isfile(os.path.join(INPUT_PATH, f))]

    # Charger les captions si dispo
    captions = {}
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
            captions = json.load(f)

    data = []
    for photo in real_photos:
        generated_path = os.path.join(INPUT_PATH, "generated_images", photo)
        if os.path.exists(generated_path):
            data.append({
                "real": os.path.join("aigen", "inputs", photo),
                "generated": os.path.join("aigen", "inputs", "generated_images", photo),
                "caption": captions.get(photo, "")
            })
    return jsonify(data)


@app.route("/file/<path:filename>")
def serve_file(filename):
    filepath = os.path.join(BASE_PATH, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return "File not found", 404


@app.route("/delete", methods=["POST"])
def delete_files():
    data = request.json
    for key in ['real', 'generated']:
        path = os.path.join(BASE_PATH, data[key])
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)

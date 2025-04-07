import gradio as gr
import numpy as np
import cv2
from deepface import DeepFace
import json
import time
import os

EMBEDDINGS_FILE = "stored_embeddings.json"

def load_embeddings():

    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            embeddings = json.load(f)
        return [np.array(embedding) for embedding in embeddings]
    except FileNotFoundError:
        return []

def save_embeddings(embeddings):
      with open(EMBEDDINGS_FILE, "w") as f:
        json.dump([embedding.tolist() for embedding in embeddings], f)

stored_embeddings = load_embeddings()

def extract_face_embedding_from_frame(frame):
    """Extract a facial embedding from a single frame (image)."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y + h, x:x + w]
        marked_frame = frame.copy()
        cv2.rectangle(marked_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        embedding = DeepFace.represent(face, model_name="Facenet")[0]["embedding"]
        return np.array(embedding), marked_frame, face

    raise Exception("No face detected in the frame.")

def verify_face_from_webcam(video, save_embedding=False, name=""):
    global stored_embeddings

    if video is None:
        return None, "Please record a video first.", None, 0

    try:
        cap = cv2.VideoCapture(video)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return None, "Error: Could not read video.", None, 0
        frame = frames[len(frames)//2]
        embedding, marked_frame, face = extract_face_embedding_from_frame(frame)
        max_similarity = 0
        for stored_embedding in stored_embeddings:
            similarity = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
            max_similarity = max(max_similarity, similarity)
            if similarity > 0.7:
                if save_embedding:
                    return marked_frame, f"⚠️ This face is already in the database. Similarity: {similarity:.2f}", face
                return marked_frame, f"✅ **AUTHENTICATED!** Similarity score: {similarity:.2f}", face, similarity

        if save_embedding:
            if not name.strip():
                return marked_frame, "⚠️ Please enter a name to save this face.", face
            metadata = {
                "embedding": embedding.tolist(),
                "name": name.strip(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            stored_embeddings.append(embedding)
            try:
                with open(EMBEDDINGS_FILE, "r") as f:
                    all_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_data = []

            all_data.append(metadata)
            with open(EMBEDDINGS_FILE, "w") as f:
                json.dump(all_data, f)

            return marked_frame, f"✅ **NEW FACE REGISTERED!** Welcome, {name}!", face

        return marked_frame, f"❌ **ACCESS DENIED!** Highest similarity: {max_similarity:.2f}", face, max_similarity

    except Exception as e:
        return None, f"⚠️ Error: {str(e)}", None, 0

def register_face(video, name=""):
    global stored_embeddings

    if video is None:
        return None, "Please record a video first.", None

    try:
        cap = cv2.VideoCapture(video)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return None, "Error: Could not read video.", None
        frame = frames[len(frames)//2]

        embedding, marked_frame, face = extract_face_embedding_from_frame(frame)

        for stored_embedding in stored_embeddings:
            similarity = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
            if similarity > 0.9:
                return marked_frame, f"⚠️ This face is already in the database.", face

        if not name.strip():
            return marked_frame, "⚠️ Please enter a name to save this face.", face
        metadata = {
            "embedding": embedding.tolist(),
            "name": name.strip(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        stored_embeddings.append(embedding)

        try:
            with open(EMBEDDINGS_FILE, "r") as f:
                all_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_data = []

        all_data.append(metadata)
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(all_data, f)

        return marked_frame, f"✅ **NEW FACE REGISTERED!** Welcome, {name}!", face

    except Exception as e:
        return None, f"⚠️ Error: {str(e)}", None

def get_registered_count():
    """Get the count of registered faces"""
    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            data = json.load(f)
        return f"Total registered faces: {len(data)}"
    except (FileNotFoundError, json.JSONDecodeError):
        return "Total registered faces: 0"

css = """
.gradio-container {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: white;
}
.title-container {
    text-align: center;
    margin-bottom: 2rem;
}
.title-container h1 {
    font-size: 2.5rem;
    background: -webkit-linear-gradient(#eee, #0e93e0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.title-container p {
    font-size: 1.2rem;
    color: #ccc;
}
.status-authenticated {
    color: #4CAF50;
    font-weight: bold;
    font-size: 1.2rem;
}
.status-denied {
    color: #F44336;
    font-weight: bold;
    font-size: 1.2rem;
}
.output-image {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}
.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #888;
}
.face-container {
    border: 2px solid #0e93e0;
    border-radius: 10px;
    padding: 10px;
    background-color: rgba(14, 147, 224, 0.1);
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
    gr.HTML("""
    <div class="title-container">
        <h1>🔐 Secure Face Authentication System</h1>
        <p>Record a video using your webcam to authenticate or register a new face</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Authentication"):
            with gr.Row():
                with gr.Column(scale=3):
                    video_input = gr.Video(label="Record a short video (1-2 seconds)", format="mp4")
                    auth_button = gr.Button("Authenticate", variant="primary")

                with gr.Column(scale=2):
                    output_image = gr.Image(label="Detection Result")
                    face_image = gr.Image(label="Extracted Face", elem_classes="output-image")

            with gr.Row():
                status_text = gr.Markdown("Ready for authentication...")
                confidence_meter = gr.Number(label="Confidence Score", value=0, minimum=0, maximum=1)

        with gr.Tab("Registration"):
            with gr.Row():
                with gr.Column(scale=3):
                    reg_video_input = gr.Video(label="Record a short video (1-2 seconds)", format="mp4")
                    name_input = gr.Textbox(label="Enter your name", placeholder="John Doe")
                    register_button = gr.Button("Register New Face", variant="secondary")

                with gr.Column(scale=2):
                    reg_output_image = gr.Image(label="Detection Result")
                    reg_face_image = gr.Image(label="Extracted Face", elem_classes="output-image")

            with gr.Row():
                reg_status_text = gr.Markdown("Ready for registration...")
    with gr.Row():
        stats_text = gr.Markdown(get_registered_count())
        refresh_button = gr.Button("Refresh Stats", variant="secondary", size="sm")

    gr.HTML("""
    <div class="footer">
        <p>Powered by DeepFace + Gradio | © 2025 Face Authentication System</p>
    </div>
    """)
    auth_button.click(
        fn=verify_face_from_webcam,
        inputs=[video_input, gr.Checkbox(value=False, visible=False)],
        outputs=[output_image, status_text, face_image, confidence_meter]
    )
    register_button.click(
        fn=register_face,
        inputs=[reg_video_input, name_input],
        outputs=[reg_output_image, reg_status_text, reg_face_image]
    )

    refresh_button.click(fn=get_registered_count, inputs=None, outputs=[stats_text])

if __name__ == "__main__":
    iface.launch()

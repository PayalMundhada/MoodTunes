import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from googleapiclient.discovery import build
import random
import gdown
import os
import keys

YOUTUBE_API_KEY = keys.YOUTUBE_API_KEY
youtube = build("youtube", "v3", developerKey= YOUTUBE_API_KEY)

def preprocess_image(input_image, target_size=(48, 48)):

    # Conevrt input image to grayscale
    gray = np.array(image.convert("L"))
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face = gray[y:y+h, x:x+w]  # Crop face region
        face = cv2.resize(face, target_size)  # Resize
        face = face / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        return face
    return None  # If no face is detected

def predict_emotion_merged(input_image):
    image = preprocess_image(input_image)
    if image is None:
        print("No face detected.")
        return

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    emotion_classes = ["Negative", "Neutral", "Positive"]
    predicted_emotion = emotion_classes[np.argmax(prediction)]
    return predicted_emotion

## function to fetch  YouTube music playlists based on detected emotion and selected language
def get_youtube_playlists(emotion, language):

    # Define multiple search queries for each emotion
    query_dict = {
        "Positive":
        [
            "feel good music", 
            "happy upbeat songs"
        ],
        "Negative": [
            "stress relief music", 
            "calm relaxing songs"
        ],
        "Neutral": [
            "Lo-Fi Beats music", 
            "chill study music"
        ]
    }

    # Ensure emotion is valid; otherwise, default to "relaxing music"
    query_options = query_dict.get(emotion, ["relaxing music"])
    query = random.choice(query_options)  # Pick a random query

    
    query += f" {language}"

    # Make an API request to search for playlists
    request = youtube.search().list(
        q=query,           # Search query
        part="snippet",    # Get playlist metadata (title, description, etc.)
        maxResults=5,      # Number of results
        type="playlist",   # Fetch only playlists
        order="viewCount"  # with maximum views
    )
    
    response = request.execute()  # Execute the API request
    return response




# Load your trained CNN model
# Google Drive File ID
file_id = "13E0cUC1uPojujg1RreJ8FqowNrxGNSOT"
# Define model path
MODEL_PATH = "mood_model.h5"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)



# MODEL_PATH = "./models/cnn_model_merged_labels.h5"  # Update with your model's filename
model = tf.keras.models.load_model(MODEL_PATH)
# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define mood labels (must match your CNN model's output labels)
mood_labels = ["Negative", "Neutral", "Positive"]


# Define mood images 
image_paths = [
    "./Dataset/happyimage.jpeg", "./Dataset/n2.jpeg", "./Dataset/sad.jpeg",
    "./Dataset/nt1.jpeg", "./Dataset/nt2.jpeg", "./Dataset/nt3.jpeg",
    "./Dataset/surprised2.jpeg", "./Dataset/p2.jpeg", "./Dataset/n4.jpeg",
    "./Dataset/angry.jpeg", "./Dataset/angry2.jpeg", "./Dataset/disgust1.jpeg",
    "./Dataset/disgust2.jpeg", "./Dataset/disgust3.jpeg", "./Dataset/fear1.jpeg",
    "./Dataset/neutral1.jpeg","./Dataset/neutral2.jpeg", "./Dataset/neutral3.jpeg",
    "./Dataset/nt4.jpeg","./Dataset/p4.jpeg", "./Dataset/sad2.jpeg",
    "./Dataset/surprised.jpeg","./Dataset/surprised1.jpeg", "./Dataset/p1.jpeg"
    
]
# Fixed size for images
IMG_SIZE = (200, 200)  # Adjust as needed

# Initialize session state for selected image
if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None

# Function to handle selection (ensuring only one selection)
def select_image(img_path):
    st.session_state["selected_image"] = img_path  # Store selected image
    # Reset all checkboxes
    for img in image_paths:
        st.session_state[f"checkbox_{img}"] = (img == img_path)

# Streamlit UI
st.title("Welcome to Mood-Tunes 🎵")
st.header("Pick a Vibe, Get a Playlist!")


# Create a 3x3 grid for images with checkboxes
cols = st.columns(4)  # Create 3 columns

for idx, img_path in enumerate(image_paths):
    with cols[idx % 4]:  # Distribute images in columns
        img = Image.open(img_path)
        img = img.resize(IMG_SIZE)
        st.image(img, use_container_width=False)
        # Use checkbox with session state
        checkbox = st.checkbox("", key=f"checkbox_{img_path}",on_change=select_image, args=(img_path,))

# Process the selected image
if st.session_state["selected_image"]:
    selected_image = st.session_state["selected_image"]
    image = Image.open(selected_image)

    # Resize image to fixed dimensions
    fixed_size = (200, 200)  # (width, height)
    resized_image = image.resize(fixed_size)

    st.header("Please select language for your playlist?")

    selected_language = st.selectbox(
    "",
    ("English", "Deutsch", "Hindi", "French", "Arabic", "Spanish", "Russian", "Punjabi", "Telgu", "Tamil" ),
    placeholder = "Select laungaue  of your playlist...",
    )

    # Predict emotion
    detected_emotion = predict_emotion_merged(image)

    st.header("You have selected")
    # Create two columns for side-by-side layout
    col_img, col_info = st.columns([1, 2])  # 1: Thumbnail, 2: Playlist info

    # Display in columns
    with col_img:
        st.image(resized_image, width=200)  # Show selected image

    with col_info:
        st.subheader(f"Mood : {detected_emotion}")  # Show playlist title & link
        st.subheader(f"Language:{selected_language}")  # Show playlist title & link
    # Recommend YouTube playlist
    youtube_response = get_youtube_playlists(detected_emotion, selected_language)

     # Print recommended playlist links
    st.header(f"Here’s Your Perfect Match! 🎶")
    for item in youtube_response['items']:
        title = item["snippet"]["title"]
        playlist_id = item["id"]["playlistId"]
        thumbnail = item["snippet"]["thumbnails"]["medium"]["url"]
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

        # Create two columns for side-by-side layout
        col1, col2 = st.columns([1, 2])  # 1: Thumbnail, 2: Playlist info

        # Display in columns
        with col1:
            st.image(thumbnail, width=100)  # Show thumbnail

        with col2:
            st.write(f"[{title}]({playlist_url})")  # Show playlist title & link

        st.write("---")  # Divider line for better UI

    
st.header("🎭 Enjoy your music!")

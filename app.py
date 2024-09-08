import streamlit as st
import cv2
import numpy as np
from yolo_utils import load_yolo, detect_objects, draw_labels
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt

# Database setup
engine = create_engine('sqlite:///users.db')
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Streamlit setup
st.set_page_config(page_title="Helmet Detection", page_icon="🛡️", layout="wide")

# Utility functions
def add_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user = User(username=username, password=hashed_password)
    session.add(user)
    session.commit()

def authenticate_user(username, password):
    user = session.query(User).filter_by(username=username).first()
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
        return True
    return False

# CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: white;
    }
    .reportview-container {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
    }
    .css-18e3th9 {
        background-color: #000000 !important;
    }
    .css-1lcbmhc {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Pages
def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success("Logged in successfully")
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
        else:
            st.error("Invalid username or password")

def signup_page():
    st.title("Sign Up")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Sign Up"):
        add_user(username, password)
        st.success("You have successfully signed up. Please log in.")

def about_page():
    st.title("About Us")
    st.write("This is a safety helmet detection app using YOLO.")
    st.write("It detects whether a person is wearing a helmet and displays the result in real-time.")

def home_page():
    st.title("Safety Helmet Detection")

    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.warning("You need to login to access this page.")
        return

    yolo_net, classes, output_layers = load_yolo()

    option = st.selectbox("Select an option", ["Upload Image", "Live Webcam"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            height, width = image.shape[:2]

            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            yolo_net.setInput(blob)
            detections = yolo_net.forward(output_layers)

            boxes, confidences, class_ids = detect_objects(detections, width, height, classes)

            for i, box in enumerate(boxes):
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)
                draw_labels(image, label, confidence, x, y, x + w, y + h, color)

            st.image(image, channels="BGR")

    elif option == "Live Webcam":
        st.write("Starting webcam...")
        cap = cv2.VideoCapture(0)

        if st.button("Stop Webcam"):
            cap.release()
            cv2.destroyAllWindows()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            yolo_net.setInput(blob)
            detections = yolo_net.forward(output_layers)

            boxes, confidences, class_ids = detect_objects(detections, width, height, classes)

            for i, box in enumerate(boxes):
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)
                draw_labels(frame, label, confidence, x, y, x + w, y + h, color)

            st.image(frame, channels="BGR")

    st.write("Choose an option to detect if a person is wearing a helmet.")

# Main page setup
st.sidebar.title("Navigation")
PAGES = {
    "Home": home_page,
    "Login": login_page,
    "Sign Up": signup_page,
    "About Us": about_page,
}

if 'page' not in st.session_state:
    st.session_state['page'] = "Login"

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
st.session_state['page'] = selection
page = PAGES[selection]
page()

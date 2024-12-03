import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import sqlite3
import hashlib
import json
from datetime import datetime
import matplotlib.pyplot as plt
import io
from datetime import date
import cv2

# Database setup
conn = sqlite3.connect('app_database.db')
cursor = conn.cursor()

# Create Users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

# Create Predictions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    upload_time TEXT NOT NULL,
    image_name TEXT NOT NULL,
    image BLOB NOT NULL,
    predicted_class TEXT NOT NULL,
    probabilities TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
)
""")

# Create Audit Log table
cursor.execute("""
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
)
""")
conn.commit()

# Load the pre-trained model
model = load_model('kidney_disease_detection_model_saujanya.keras')

# Define class names
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Streamlit App
st.title("Automated Kidney Disease Detection in CT Images Using Convolutional Neural Networks (CNN) model")

# Login/Register functionality
menu = st.sidebar.selectbox("Menu", ["Login", "Register"])

# Set the minimum and maximum dates
min_date = date(1900, 1, 1)  # Earliest selectable date
max_date = date.today()      # Latest selectable date (today)

if menu == "Register":
    st.subheader("Create a New Account")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    dob = st.date_input("Date of Birth", value=date(2000, 1, 1), min_value=min_date, max_value=max_date)

    if st.button("Register"):
        if first_name and last_name and email and password:
            try:
                hashed_password = hash_password(password)
                cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                               (f"{first_name} {last_name}", email, hashed_password))
                conn.commit()
                st.success("Registration successful! Please log in.")
            except sqlite3.IntegrityError:
                st.error("Email already exists. Try logging in.")
        else:
            st.error("Please fill out all fields.")

if menu == "Login":
    st.subheader("Login to Your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed_password = hash_password(password)
        cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, hashed_password))
        user = cursor.fetchone()

        if user:
            st.success(f"Welcome {user[1]}!")
            st.session_state["user_id"] = user[0]
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid email or password.")

# Application functionality after login
if st.session_state.get("logged_in", False):
    st.write("Upload a CT scan image of the kidney to classify whether it shows a Stone, Tumor, Cyst, or Normal kidney.")

    # File upload widget
    uploaded_file = st.file_uploader("Choose a CT Scan image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_data = uploaded_file.read()
        img = image.load_img(io.BytesIO(img_data), target_size=(150, 150), color_mode='grayscale')  # Resize and convert to grayscale
        img_array = image.img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        try:
            # Make the prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            # Display the result
            st.write(f"Prediction: {class_names[predicted_class]}")
            
            # Display prediction probability
            st.write(f"Prediction probability: {prediction[0][predicted_class] * 100:.2f}%")
            probabilities_json = json.dumps(prediction[0].tolist())
            st.write("Class probabilities:")
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")

            st.subheader('Predicted Class Probability')
            fig, ax = plt.subplots()
            ax.bar(class_names, prediction[0], color='royalblue')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Probability')
            ax.set_title('Class Probability Distribution')
            st.pyplot(fig)

            # Save prediction details to the database
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO predictions (user_id, upload_time, image_name, image, predicted_class, probabilities)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (st.session_state["user_id"], upload_time, uploaded_file.name, img_data, class_names[predicted_class], probabilities_json))
            conn.commit()

            # Log the action
            cursor.execute("""
                INSERT INTO audit_logs (user_id, action, timestamp)
                VALUES (?, ?, ?)
            """, (st.session_state["user_id"], "Uploaded Image and Predicted", upload_time))
            conn.commit()
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Option to logout
    if st.button("Logout"):
        # Log the user out by setting logged_in to False
        st.session_state["logged_in"] = False
        st.rerun()

else:
    # If not logged in, show login/registration functionality (assuming you have it elsewhere in the code)
    st.write("Please log in to use the app.")
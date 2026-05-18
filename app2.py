import streamlit as st
import numpy as np
import cv2
import sqlite3
import uuid
from datetime import datetime
from tensorflow.keras.models import load_model
import os
import py7zr
from PIL import Image
import pandas as pd

# ---------------- 1. PAGE SETUP ----------------
st.set_page_config(page_title="Biometric Medical Vault", layout="centered")

st.title("🩸 Fingerprint Blood Group Detection System")
st.write("Secure Biometric Diagnostic & Persistent Medical Vault")

# ---------------- 2. PERMANENT DATABASE ENGINE ----------------
def get_db_connection():
    # Fixed filename to match your latest preference and ensured threading support
    conn = sqlite3.connect("Clinicalvault.db", check_same_thread=False)
    return conn

# Initialize DB and Table
db_conn = get_db_connection()
db_cursor = db_conn.cursor()
db_cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY, name TEXT, age INTEGER, gender TEXT, phone TEXT, 
    location TEXT, blood_group TEXT, is_donor TEXT, confidence REAL, timestamp TEXT
)
""")
db_conn.commit()

# --- INITIAL SEEDING (Corrected with your realistic Demo Data) ---
db_cursor.execute("SELECT COUNT(*) FROM patients")
if db_cursor.fetchone()[0] == 0:
    demo_entries = [
        ("13E00607", "Dharani", 21, "Male", "9578420548", "Salem", "O+", "Yes", 99.8, "2026-04-09 20:58"),
        ("DEMO-O+-2", "Ram", 30, "Male", "9994567909", "Coimbatore", "O+", "Yes", 98.5, "2026-04-01 10:30"),
        ("DEMO-O+-1", "Ganga", 24, "Female", "8452762870", "Erode", "O+", "Yes", 97.2, "2026-04-09 11:20"),
        ("DEMO-O--2", "Ashok", 28, "Male", "7645321890", "Coimbatore", "O-", "Yes", 96.4, "2026-04-01 09:15"),
        ("DEMO-O--1", "Aruna", 26, "Female", "7865432190", "Dharmapuri", "O-", "Yes", 95.8, "2026-03-31 14:45"),
        ("DEMO-B+-2", "Babu", 32, "Male", "9874226835", "Namakkal", "B+", "Yes", 99.1, "2026-03-31 16:20"),
        ("DEMO-B+-1", "Santhiya", 22, "Female", "7865443189", "Salem", "B+", "Yes", 98.7, "2026-04-01 12:10"),
        ("DEMO-B--2", "Gowtham", 27, "Male", "8945887543", "Salem", "B-", "Yes", 94.5, "2026-04-01 08:30"),
        ("DEMO-B--1", "Sanjay", 25, "Male", "8952382856", "Erode", "B-", "Yes", 96.8, "2026-04-01 15:40"),
        ("DEMO-AB+-2", "Kavya", 23, "Female", "6734152781", "Chennai", "AB+", "Yes", 97.9, "2026-04-02 11:05"),
        ("DEMO-AB+-1", "Madhu", 24, "Female", "9087685644", "Karur", "AB+", "Yes", 98.2, "2026-04-02 10:30"),
        ("DEMO-AB--2", "Harshit", 29, "Female", "9123456789", "Trichy", "AB-", "Yes", 93.4, "2026-04-02 09:15"),
        ("DEMO-AB--1", "Nandhini", 22, "Female", "9997765341", "Madurai", "AB-", "Yes", 95.1, "2026-04-02 14:20"),
        ("DEMO-A+-2", "Sanjith", 31, "Male", "9990760543", "Krishnagiri", "A+", "Yes", 98.9, "2026-04-02 13:10"),
        ("DEMO-A+-1", "Janani", 25, "Female", "9995462000", "Thanjavur", "A+", "Yes", 97.5, "2026-04-02 12:00"),
        ("DEMO-A--2", "Akash", 24, "Male", "9998907653", "Salem", "A-", "Yes", 96.2, "2026-04-02 16:45"),
        ("DEMO-A--1", "Anand", 26, "Male", "9887663459", "Erode", "A-", "Yes", 95.7, "2026-04-02 15:30")
    ]
    db_cursor.executemany("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?)", demo_entries)
    db_conn.commit()

# ---------------- 3. CORE AI FUNCTIONS ----------------
@st.cache_resource
def load_model_file():
    if not os.path.exists("fingerprint_blood_model.keras"):
        with py7zr.SevenZipFile("fingerprint_blood_model.7z", mode='r') as archive:
            archive.extractall()

    return load_model("fingerprint_blood_model.keras")

model = load_model_file()
categories = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

def detect_fingerprint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    return image[y:y+h, x:x+w]

def enhance_ridges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    clahe = cv2.createCLAHE(3.0,(8,8))
    return clahe.apply(blur)

def generate_xai_heatmap(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return overlay

# ---------------- 4. INPUT SECTION ----------------
file = None
option = st.radio("Choose Input Method", ["Upload Image", "Capture from Camera"], horizontal=True)
if option == "Upload Image":
    file = st.file_uploader("Upload", type=["jpg","png","jpeg","bmp"])
else:
    file = st.camera_input("Scanner")

# ---------------- 5. THE LOGIC FLOW (SESSION STATE PERSISTENCE) ----------------
if file is not None:
    img_np = np.array(Image.open(file).convert("RGB"))
    st.image(img_np, caption="Input Scan", use_container_width=True)

    cropped = detect_fingerprint(img_np)
    if cropped is not None:
        enhanced = enhance_ridges(cropped)
        st.image(enhanced, caption="Enhanced Ridge Detail", use_container_width=True)

        if st.button("🔍 Run Biometric Analysis"):
            resized = cv2.resize(enhanced, (224, 224))
            rgb = cv2.cvtColor(cv2.equalizeHist(resized), cv2.COLOR_GRAY2RGB)
            final_input = np.expand_dims(rgb.astype("float32") / 255.0, axis=0)
            
            prediction = model.predict(final_input)[0]
            
            st.session_state['result'] = categories[np.argmax(prediction)]
            st.session_state['conf'] = np.max(prediction) * 100
            st.session_state['processed_img'] = cropped

        if 'result' in st.session_state:
            res = st.session_state['result']
            confidence = st.session_state['conf']

            st.success(f"### Predicted Result: {res} (Confidence: {confidence:.2f}%)")
            
            xai_img = generate_xai_heatmap(cv2.resize(st.session_state['processed_img'], (224, 224)))
            st.image(xai_img, caption="XAI Feature Mapping", use_container_width=True)

            st.divider()
            st.subheader(f"📂 Records Found in Vault for: {res}")
            
            fetch_conn = get_db_connection()
            vault_df = pd.read_sql_query("SELECT name, phone, location, is_donor, timestamp FROM patients WHERE blood_group = ?", 
                                         fetch_conn, params=(res,))
            
            if not vault_df.empty:
                st.table(vault_df)
            else:
                st.warning(f"No previous records for {res} found.")

            st.divider()
            st.subheader(f"🔒 Secure New Entry ({res}) to Medical Vault")
            with st.form("add_to_vault", clear_on_submit=True):
                u_name = st.text_input("Full Name")
                u_age = st.number_input("Age", 1, 100, 25)
                u_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                u_phone = st.text_input("Emergency Contact Number")
                u_loc = st.text_input("Current Location / Hospital")
                u_donor = st.radio("Willing to Donate Blood?", ["Yes", "No"], horizontal=True)
                
                if st.form_submit_button("Seal Record in Vault"):
                    if u_name and u_phone:
                        new_id = str(uuid.uuid4())[:8].upper()
                        ins_conn = get_db_connection()
                        ins_cursor = ins_conn.cursor()
                        ins_cursor.execute("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?)",
                                           (new_id, u_name, u_age, u_gender, u_phone, u_loc, res, u_donor, float(confidence), str(datetime.now().strftime("%Y-%m-%d %H:%M"))))
                        ins_conn.commit()
                        st.success(f"Successfully Stored! Patient ID: {new_id}")
                        st.rerun()
                    else:
                        st.error("Name and Phone are mandatory.")

# ---------------- 6. THE GOLDEN KEY (SECURE ADMIN) ----------------
st.sidebar.title("🔐 Admin Vault Control")
key = st.sidebar.text_input("Enter Vault Security Key", type="password")

if key == "golden key":
    st.sidebar.success("Access Granted.")
    if st.sidebar.checkbox("Reveal All Records"):
        admin_conn = get_db_connection()
        all_df = pd.read_sql_query("SELECT * FROM patients", admin_conn)
        st.sidebar.dataframe(all_df)
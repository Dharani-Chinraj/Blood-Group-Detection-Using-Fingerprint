# Blood-Group-Detection-Using-Fingerprint

To set up a professional-grade GitHub repository for your final year project, you need a clear `README.md` and a clean upload process.

### **Part 1: GitHub Project Description (README.md)**

Copy and paste this into your repository. It highlights the technical depth of your **Vision Transformer (ViT)** and **DID Security** modules.

---

# **Guardian-ID: Non-Invasive Blood Group Detection & Secure Medical Vault**

### **Project Overview**
Guardian-ID is a cutting-edge medical diagnostic tool that predicts blood groups using touchless fingerprint acquisition. By utilizing **Vision Transformers (ViT)** and **Dermatoglyphic** science, the system eliminates the need for chemical reagents and invasive needle pricks.

### **Core Modules**
* **Module 1-3: AI Engine:** A Vision Transformer architecture that analyzes ridge density and patterns (Loops, Whorls, Arches) to identify blood group correlations.
* **Module 4: Mobile Deployment:** A responsive **Streamlit** application optimized for smartphone cameras, featuring secure image transmission.
* **Module 5-6: Decentralized Medical Vault:** Implementation of **DIDs (Decentralized Identifiers)** to ensure patient data is securely locked and accessible only via biometric verification.

### **Tech Stack**
* **Language:** Python
* **Deep Learning:** Hugging Face Transformers (ViT), PyTorch/TensorFlow
* **Computer Vision:** OpenCV (`cv2`)
* **Frontend/Deployment:** Streamlit
* **Environment:** Jupyter Notebook / Google Colab

---

### **Part 2: How to Upload to GitHub**

Follow these steps to upload your code and create a "workable" link.

**Step 1: Create the Repository**
1. Go to [GitHub](https://github.com/) and click **New Repository**.
2. Name it (e.g., `Guardian-ID-Blood-Group-Detection`).
3. Set it to **Public** and check the box to **Add a README file**.

**Step 2: Upload Your Files**
1. Click **Add file** -> **Upload files**.
2. Drag and drop your `.py` files (Streamlit app), your `.ipynb` (Jupyter Notebook), and your trained model files (`.pth` or `.h5`).
3. **Critical:** Create a file named `requirements.txt` and list your libraries (e.g., `streamlit`, `opencv-python`, `transformers`, `torch`).

**Step 3: Create the "Workable Link" (Deployment)**
Since you are using Streamlit, the best way to give a "workable link" is through **Streamlit Cloud**:
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Connect your GitHub account.
3. Select your repository and the main Python file (e.g., `app.py`).
4. Click **Deploy**.
5. Streamlit will give you a link (e.g., `guardian-id.streamlit.app`). **Paste this link into the "Website" section of your GitHub repository settings.**

---

### **Quick Checklist before you upload:**
* **Remove API Keys:** Ensure no private passwords or keys are in your code.
* **Sample Images:** Include a folder called `samples/` with a few blurred fingerprint images so others can test the app.
* **License:** Add an **MIT License** (it tells people they can look at your code but you own the project).

Now you can share the GitHub link and the Streamlit link with your external examiners!

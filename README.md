# RetinaAI

### Explainable Diabetic Retinopathy Detection Using Deep Learning

RetinaAI is an AI-powered retinal image analysis application that uses deep learning to classify diabetic retinopathy severity from retinal fundus images.

The application uses a DenseNet-121 model to classify retinal images into five diabetic retinopathy severity levels and uses Grad-CAM to provide a visual explanation of the model's prediction.

---

## 🚀 Features

- Retinal fundus image upload
- AI-based diabetic retinopathy classification
- DenseNet-121 deep learning model
- Five-class severity classification
- Prediction confidence score
- Class probability visualization
- Grad-CAM explainability
- Interactive Streamlit interface
- CPU/GPU device detection
- Real-time image inference

---

## 🧠 Diabetic Retinopathy Classes

The model predicts five severity levels:

| Class | Severity |
|---|---|
| 0 | No Diabetic Retinopathy |
| 1 | Mild Diabetic Retinopathy |
| 2 | Moderate Diabetic Retinopathy |
| 3 | Severe Diabetic Retinopathy |
| 4 | Proliferative Diabetic Retinopathy |

---

## 🏗️ Architecture

The application follows this pipeline:

```text
Retinal Fundus Image
        │
        ▼
Image Preprocessing
        │
        ▼
224 × 224 Image
        │
        ▼
DenseNet-121
        │
        ▼
5-Class Classification
        │
        ├───────────────┐
        ▼               ▼
Prediction         Class Probabilities
        │
        ▼
   Grad-CAM
        │
        ▼
Visual Explanation
```

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- DenseNet-121
- Grad-CAM
- Streamlit
- NumPy
- Pillow

---

## 📁 Project Structure

```text
RetinaAI/
│
├── app.py
├── best_densenet121.pth
├── requirements.txt
├── README.md
├── .gitignore
├── DataLoading.ipynb
│
└── .streamlit/
    └── config.toml
```

---

## 🤖 Model

RetinaAI uses a trained DenseNet-121 model for five-class diabetic retinopathy classification.

The application expects the trained model checkpoint:

```text
best_densenet121.pth
```

The model file must be located in the project root:

```text
RetinaAI/
│
├── app.py
└── best_densenet121.pth
```

The application loads this checkpoint during startup and uses it for inference.

---

## 🖼️ Image Preprocessing

Uploaded retinal images are processed before being passed to the model.

The preprocessing pipeline includes:

1. Convert the image to RGB.
2. Resize the image to 224 × 224 pixels.
3. Convert the image to a PyTorch tensor.
4. Apply ImageNet normalization.
5. Pass the processed image through DenseNet-121.
6. Calculate class probabilities using softmax.
7. Display the predicted class and confidence.
8. Generate a Grad-CAM visualization.

### ImageNet Normalization

```text
Mean:
[0.485, 0.456, 0.406]

Standard Deviation:
[0.229, 0.224, 0.225]
```

---

## 🔍 Explainable AI with Grad-CAM

RetinaAI uses Grad-CAM to provide an interpretable visualization of the model's prediction.

The Grad-CAM heatmap highlights regions of the retinal image that contributed to the model's prediction.

This provides additional insight into the model's decision instead of displaying only the predicted class.

```text
Input Retinal Image
        │
        ▼
   DenseNet-121
        │
        ▼
   Prediction
        │
        ▼
    Grad-CAM
        │
        ▼
Attention Heatmap
```

---

## 📊 Training

The project includes the training notebook:

```text
DataLoading.ipynb
```

The notebook contains the workflow for:

- Dataset loading
- Data exploration
- Train/test splitting
- Image preprocessing
- Data augmentation
- Dataset creation
- DataLoader creation
- CNN experimentation
- ResNet experimentation
- EfficientNet experimentation
- DenseNet experimentation
- Model evaluation

The training workflow uses the APTOS 2019 Blindness Detection dataset.

---

## 📚 Dataset

The model development workflow uses the:

**APTOS 2019 Blindness Detection Dataset**

The complete dataset is required only during model training.

The dataset is **not required for running the deployed Streamlit application**.

Only the trained model checkpoint is required for inference.

---

## 💻 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project directory:

```bash
cd RetinaAI
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

Activate the environment:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deployment on Streamlit Community Cloud

RetinaAI can be deployed directly from GitHub using Streamlit Community Cloud.

### Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the `RetinaAI` repository.
5. Select the branch containing the project.
6. Set the main file to:

```text
app.py
```

7. Deploy the application.
8. Wait for dependencies to install.
9. Open the generated application URL.

---

## 📦 Deployment Requirements

The Streamlit deployment requires:

```text
app.py
requirements.txt
best_densenet121.pth
```

The following files are recommended for the GitHub repository:

```text
README.md
.gitignore
DataLoading.ipynb
.streamlit/config.toml
```

---

## ⚠️ Model File

The application requires:

```text
best_densenet121.pth
```

If this file is too large for GitHub, the model should be hosted using an appropriate external model-storage service and downloaded by the application when required.

Do not upload the complete training dataset to the deployment repository.

---

## 🔒 Security

Do not upload:

- API keys
- Passwords
- Access tokens
- `.env` files
- Kaggle credentials
- Personal credentials

Sensitive information should be stored using environment variables or Streamlit secrets.

---

## 📌 Important Notes

The training dataset and training artifacts are not required for inference.

The deployed application only needs the trained model and the dependencies required by the Streamlit application.

---

## 🔮 Future Improvements

Potential improvements include:

- Improved model accuracy
- Larger and more diverse datasets
- Better retinal image preprocessing
- Model calibration
- Additional explainability techniques
- Model versioning
- External model storage
- Faster inference
- Cloud-based model management
- Clinical validation
- Authentication and user management

---

## ⚕️ Disclaimer

RetinaAI is an educational and research-oriented AI project.

It is **not a medical diagnostic device**.

Predictions generated by this application should not be considered a medical diagnosis and should not replace evaluation by a qualified ophthalmologist or other healthcare professional.

---

## 👨‍💻 Author

**Aaditya Hole**

---

## 📄 License

Add an appropriate open-source license before publicly distributing the project.

Also review the licensing and usage terms of the dataset and model components before redistributing them.
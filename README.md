🌱 Agri-Spectral Analyst: Hyperspectral Crop Classification
📝 Overview
A Streamlit web application for agricultural hyperspectral image analysis that classifies crop types using optimized 3D CNN models. The tool provides crop distribution visualization, field health assessment, and actionable farming recommendations with fast processing times.

✨ Features
Three model intensity options (Fast/Balanced/Full)

Interactive field visualization with PCA

Crop distribution analysis

Real-time training progress monitoring

Field health recommendations

Confusion matrix and accuracy metrics

⚙️ Installation
Prerequisites
Python 3.8+

pip package manager

Step-by-Step Setup
Clone the repository

bash
git clone https://github.com/yourusername/agri-spectral-analyst.git
cd agri-spectral-analyst
Create and activate virtual environment (Recommended)

bash
python -m venv agri-env
source agri-env/bin/activate  # Linux/Mac
.\agri-env\Scripts\activate  # Windows
Install dependencies

bash
pip install -r requirements.txt
🚀 Usage
Run the application

bash
streamlit run hyperspectral_streamlit.py
In your browser

Upload hyperspectral data (.npy) and label files (.npy)

Configure analysis parameters in sidebar

View real-time training progress

Analyze results and recommendations

📂 File Structure
agri-spectral-analyst/
├── hyperspectral_streamlit.py  # Main application
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── sample_data/               # Example datasets
    ├── field_data.npy
    └── labels.npy
📊 Sample Data
Example datasets are available in the sample_data directory for testing:

field_data.npy: Hyperspectral data cube

labels.npy: Ground truth crop labels

🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request with your improvements.

📜 License
This project is licensed under the MIT License.

# 🌱 Agri-Spectral Analyst

## 📝 Overview
A Streamlit web app for hyperspectral crop classification using 3D CNNs. Processes agricultural field data, visualizes crop distribution, and provides health assessments with farming recommendations.

## ✨ Features
- Three model modes (Fast/Balanced/Precision)
- Interactive field visualizations
- Real-time training monitoring
- Crop health analytics
- Performance metrics

## ⚙️ Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/agri-spectral-analyst.git
cd agri-spectral-analyst
python -m venv agri-env
source agri-env/bin/activate  # Linux/Mac
.\agri-env\Scripts\activate  # Windows
pip install streamlit numpy matplotlib scikit-learn seaborn tensorflow pandas pillow
streamlit run hyperspectral_streamlit.py
agri-spectral-analyst/
├── hyperspectral_streamlit.py  # Main app
├── requirements.txt           # Dependencies
├── README.md
└── sample_data/               # Test datasets
    ├── field_data.npy
    └── labels.npy

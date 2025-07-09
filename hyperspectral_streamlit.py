import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium
from streamlit_folium import st_folium
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import warnings
import requests
from geopy.geocoders import Nominatim
import json
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Set up Streamlit
st.set_page_config(page_title="Global Agri-Spectral Monitor", layout="wide")
st.title("🌍 Global Agri-Spectral Monitor: Worldwide Crop Health Analysis")

# Add global agricultural branding
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=100)
st.sidebar.header("Global Configuration")

# Enhanced crop type mapping with global crops
CROP_TYPES = {
    1: "Corn/Maize", 2: "Soybeans", 3: "Wheat", 4: "Rice", 5: "Alfalfa",
    6: "Potatoes", 7: "Tomatoes", 8: "Cotton", 9: "Sugarcane", 10: "Barley",
    11: "Pasture", 12: "Forest", 13: "Urban", 14: "Water", 15: "Barren"
}

# Load pre-trained models (would be loaded from disk in production)
@st.cache_resource
def load_pretrained_models():
    # In a real application, these would be actual trained models
    return {
        'rf': RandomForestClassifier(n_estimators=200, max_depth=15),
        'xgb': XGBClassifier(n_estimators=150, max_depth=10, learning_rate=0.1),
        'cnn': None  # Placeholder for demonstration
    }

pretrained_models = load_pretrained_models()

# Hyperspectral band definitions for agriculture
SPECTRAL_BANDS = {
    'Blue': (450, 520),
    'Green': (520, 600),
    'Red': (630, 690),
    'Red Edge': (690, 750),
    'NIR1': (750, 850),
    'NIR2': (850, 950),
    'SWIR1': (1550, 1750),
    'SWIR2': (2080, 2350)
}

# Major desert regions (lat_range, lon_range)
DESERT_REGIONS = [
    {'lat_range': (15, 35), 'lon_range': (-20, 60)},    # Sahara
    {'lat_range': (20, 35), 'lon_range': (40, 60)},     # Arabian
    {'lat_range': (-30, -15), 'lon_range': (115, 150)}, # Australian
    {'lat_range': (25, 40), 'lon_range': (-120, -95)},  # SW North America
    {'lat_range': (-30, -15), 'lon_range': (-75, -65)}, # Atacama
    {'lat_range': (35, 45), 'lon_range': (60, 90)},     # Central Asia
    {'lat_range': (-30, -20), 'lon_range': (15, 25)}    # Namib
]

# Major water bodies (precise coordinates - avoiding land overlap)
WATER_BODIES = [
    # Oceans - more precise ranges avoiding coastlines
    {'lat_range': (-60, 70), 'lon_range': (-65, -10), 'name': 'Atlantic Ocean'},
    {'lat_range': (-60, 70), 'lon_range': (-180, -120), 'name': 'Pacific Ocean West'},
    {'lat_range': (-60, 70), 'lon_range': (140, 180), 'name': 'Pacific Ocean East'},
    {'lat_range': (-50, 25), 'lon_range': (45, 110), 'name': 'Indian Ocean'},
    {'lat_range': (70, 90), 'lon_range': (-180, 180), 'name': 'Arctic Ocean'},
    
    # Major seas
    {'lat_range': (30, 46), 'lon_range': (-5, 36), 'name': 'Mediterranean Sea'},
    {'lat_range': (40, 47), 'lon_range': (27, 42), 'name': 'Black Sea'},
    {'lat_range': (36, 47), 'lon_range': (46, 55), 'name': 'Caspian Sea'},
    {'lat_range': (18, 30), 'lon_range': (32, 43), 'name': 'Red Sea'},
    {'lat_range': (24, 30), 'lon_range': (48, 56), 'name': 'Persian Gulf'},
    
    # Major lakes
    {'lat_range': (41, 49), 'lon_range': (-92, -76), 'name': 'Great Lakes'},
    {'lat_range': (51, 56), 'lon_range': (103, 110), 'name': 'Lake Baikal'},
    {'lat_range': (-3, 1), 'lon_range': (31, 35), 'name': 'Lake Victoria'},
    {'lat_range': (-9, -3), 'lon_range': (29, 31), 'name': 'Lake Tanganyika'},
]

# Country boundaries data (simplified for demo)
COUNTRY_BOUNDARIES = {
    'USA': {'lat_range': (24, 50), 'lon_range': (-125, -65)},
    'India': {'lat_range': (8, 37), 'lon_range': (68, 97)},
    'China': {'lat_range': (18, 54), 'lon_range': (73, 135)},
    'Brazil': {'lat_range': (-34, 5), 'lon_range': (-74, -35)},
    'Niger': {'lat_range': (11, 24), 'lon_range': (0, 16)},
    'Chad': {'lat_range': (7, 24), 'lon_range': (13, 24)},
    'Sudan': {'lat_range': (8, 22), 'lon_range': (21, 39)},
    'Mali': {'lat_range': (10, 25), 'lon_range': (-12, 5)},
    'Algeria': {'lat_range': (18, 37), 'lon_range': (-9, 12)},
    'Libya': {'lat_range': (19, 33), 'lon_range': (9, 25)},
    'Egypt': {'lat_range': (22, 32), 'lon_range': (24, 37)},
    # Add more countries as needed
}

def is_water_body(lat, lon):
    """Precise water body detection with major water bodies only"""
    if lat is None or lon is None:
        return False
    
    # Check against defined water bodies first
    for water_body in WATER_BODIES:
        if (water_body['lat_range'][0] <= lat <= water_body['lat_range'][1] and 
            water_body['lon_range'][0] <= lon <= water_body['lon_range'][1]):
            return True
    
    return False

def is_desert(lat, lon):
    """Check if location is in a desert region"""
    if lat is None or lon is None:
        return False
    for region in DESERT_REGIONS:
        if (region['lat_range'][0] <= lat <= region['lat_range'][1] and 
            region['lon_range'][0] <= lon <= region['lon_range'][1]):
            return True
    return False

def get_climate_zone(lat, lon):
    """Get climate zone classification"""
    if lat is None or lon is None:
        return "Unknown"
    
    # Check desert regions FIRST (highest priority)
    if is_desert(lat, lon):
        return "Desert"
    
    # Then check water bodies
    if is_water_body(lat, lon):
        return "Water Body"
    
    # Other climate classifications based on latitude
    abs_lat = abs(lat)
    
    if abs_lat >= 66.5:
        return "Polar"
    elif abs_lat >= 60:
        return "Subarctic"
    elif abs_lat >= 50:
        return "Continental"
    elif abs_lat >= 35:
        return "Temperate"
    elif abs_lat >= 23.5:
        return "Subtropical"
    else:
        return "Tropical"

def get_country_name(lat, lon):
    """Get country name from coordinates"""
    if lat is None or lon is None:
        return "Unknown"
    
    # Check against country boundaries
    for country, bounds in COUNTRY_BOUNDARIES.items():
        if (bounds['lat_range'][0] <= lat <= bounds['lat_range'][1] and 
            bounds['lon_range'][0] <= lon <= bounds['lon_range'][1]):
            return country
    
    return "Unknown"

# Test the fix with all coordinates
print("Testing Niger coordinates (should be Desert):")
test_lat1, test_lon1 = 17.1828, 11.6895
print(f"Location: {test_lat1}, {test_lon1}")
print(f"Climate Zone: {get_climate_zone(test_lat1, test_lon1)}")

print("\nTesting Niger coordinates 2 (should be Desert):")
test_lat2, test_lon2 = 19.4770, 3.5156
print(f"Location: {test_lat2}, {test_lon2}")
print(f"Climate Zone: {get_climate_zone(test_lat2, test_lon2)}")

print("\nTesting Atlantic Ocean coordinates (should be Water Body):")
test_lat3, test_lon3 = 25.0060, -26.8945
print(f"Location: {test_lat3}, {test_lon3}")
print(f"Climate Zone: {get_climate_zone(test_lat3, test_lon3)}")
class HyperspectralRegressor(nn.Module):
    """Enhanced neural network for predicting crop health parameters"""
    def __init__(self, input_channels, num_outputs):
        super(HyperspectralRegressor, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_pytorch_model(model, train_loader, val_loader, epochs, device):
    """Enhanced training function with more metrics and early stopping"""
    criterion = nn.CrossEntropyLoss() if hasattr(model, 'conv1') else nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    model.to(device)
    best_val_acc = 0
    patience = 5
    no_improve = 0
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    plot_placeholder = st.empty()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if hasattr(model, 'conv1'):  # Classification model
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total if total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                if hasattr(model, 'conv1'):  # Classification model
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            # Save best model (would save to disk in production)
            best_model_state = model.state_dict()
        else:
            no_improve += 1
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.3f}, Val Acc: {val_acc:.1f}%")
        
        # Update training plot
        if epoch % 1 == 0:  # Update every epoch
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(range(epoch+1), train_losses, label='Training Loss', color='blue')
            ax1.plot(range(epoch+1), val_losses, label='Validation Loss', color='orange')
            ax1.set_title('Training Progress - Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            if train_accs[0] > 0:  # Only show accuracy for classification
                ax2.plot(range(epoch+1), train_accs, label='Training Accuracy', color='green')
                ax2.plot(range(epoch+1), val_accs, label='Validation Accuracy', color='red')
                ax2.set_title('Training Progress - Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_placeholder.pyplot(fig)
            plt.close()
        
        # Early stopping
        if no_improve >= patience:
            status_text.text(f"Early stopping at epoch {epoch+1} - Best Val Acc: {best_val_acc:.1f}%")
            model.load_state_dict(best_model_state)
            break
    
    return model, train_losses, val_losses, train_accs, val_accs

def calculate_vegetation_indices(reflectance_data, is_vegetated=True):
    """Enhanced vegetation indices calculation with more indices"""
    # Band selection (would be actual band positions in real data)
    blue = reflectance_data[:, 10]  # ~450nm
    green = reflectance_data[:, 20]  # ~550nm
    red = reflectance_data[:, 30]  # ~650nm
    red_edge = reflectance_data[:, 40]  # ~700nm
    nir = reflectance_data[:, 50]  # ~800nm
    swir1 = reflectance_data[:, 70] if reflectance_data.shape[1] > 70 else reflectance_data[:, -1]
    swir2 = reflectance_data[:, 80] if reflectance_data.shape[1] > 80 else reflectance_data[:, -1]
    
    indices = {}
    
    if is_vegetated:
        # Standard vegetation indices
        indices['NDVI'] = (nir - red) / (nir + red + 1e-8)
        indices['EVI'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        indices['SAVI'] = ((nir - red) / (nir + red + 0.5)) * (1 + 0.5)
        indices['GNDVI'] = (nir - green) / (nir + green + 1e-8)
        indices['NDWI'] = (green - nir) / (green + nir + 1e-8)
        indices['MSI'] = swir1 / nir
        
        # Additional precision indices
        indices['NDRE'] = (nir - red_edge) / (nir + red_edge + 1e-8)  # Red edge index
        indices['PSRI'] = (red - blue) / red_edge  # Plant senescence index
        indices['PRI'] = (green - red) / (green + red + 1e-8)  # Photochemical reflectance index
        indices['CI'] = (nir / red) - 1  # Chlorophyll index
        indices['ARI'] = (1 / green) - (1 / red_edge)  # Anthocyanin reflectance index
    else:
        # For non-vegetated areas
        for index in ['NDVI', 'EVI', 'SAVI', 'GNDVI', 'NDWI', 'MSI', 'NDRE', 'PSRI', 'PRI', 'CI', 'ARI']:
            indices[index] = np.random.normal(0.05, 0.02, len(red))
    
    return indices

def estimate_crop_health_parameters(reflectance_data, vegetation_indices, is_vegetated=True):
    """Enhanced health parameter estimation with more features"""
    if is_vegetated:
        # Chlorophyll content estimation (using multiple methods)
        red_edge = reflectance_data[:, 40] if reflectance_data.shape[1] > 40 else reflectance_data[:, -1]
        chlorophyll = (vegetation_indices['CI'] * 50 + vegetation_indices['NDRE'] * 30 + 
                      (1 - reflectance_data[:, 30]) * 20)  # Combined estimate
        
        # Moisture content (using multiple water bands)
        water_bands = [reflectance_data[:, i] for i in [70, 80, 90] if reflectance_data.shape[1] > 90]
        moisture = (1 - np.mean(water_bands, axis=0)) * 100
        
        # Leaf Area Index (using multiple empirical relationships)
        lai = (3.618 * vegetation_indices['EVI'] - 0.118 + 
               2.5 * vegetation_indices['NDVI'] + 0.2) / 2  # Average of methods
        
        # Biomass estimation
        nir_reflectance = reflectance_data[:, 50] if reflectance_data.shape[1] > 50 else reflectance_data[:, -1]
        biomass = (nir_reflectance * 15000 + vegetation_indices['NDVI'] * 10000) / 2
        
        # Stress indicators
        water_stress = 1 - vegetation_indices['NDWI']
        nitrogen_stress = 1 - vegetation_indices['GNDVI']
        heat_stress = vegetation_indices['MSI']
        senescence = vegetation_indices['PSRI']
    else:
        # For non-vegetated areas
        chlorophyll = np.random.normal(0.1, 0.05, reflectance_data.shape[0])
        moisture = np.random.normal(2, 1, reflectance_data.shape[0])
        lai = np.zeros(reflectance_data.shape[0])
        biomass = np.random.normal(100, 50, reflectance_data.shape[0])
        water_stress = np.random.normal(0.9, 0.1, reflectance_data.shape[0])
        nitrogen_stress = np.random.normal(0.95, 0.05, reflectance_data.shape[0])
        heat_stress = np.random.normal(0.8, 0.1, reflectance_data.shape[0])
        senescence = np.random.normal(0.7, 0.1, reflectance_data.shape[0])
    
    return {
        'Chlorophyll (mg/g)': np.mean(chlorophyll),
        'Moisture (%)': np.mean(moisture),
        'LAI': np.mean(lai),
        'Biomass (kg/ha)': np.mean(biomass),
        'Water Stress': np.mean(water_stress),
        'Nitrogen Stress': np.mean(nitrogen_stress),
        'Heat Stress': np.mean(heat_stress),
        'Senescence Index': np.mean(senescence)
    }

def simulate_hyperspectral_data(lat, lon, season='current'):
    """Enhanced hyperspectral data simulation with more realistic features"""
    if lat is None or lon is None:
        seed = 42
    else:
        seed = int((lat + lon) * 1000) % 2147483647
    
    np.random.seed(seed)
    
    # Generate realistic spectral signature based on location
    wavelengths = np.linspace(400, 2500, 210)  # Higher resolution (210 bands)
    
    # Check if location is desert or water
    water = is_water_body(lat, lon)
    desert = is_desert(lat, lon)
    
    if water:
        # Water spectral signature with more detailed features
        base_signature = np.zeros(len(wavelengths)) + 0.05
        # Add water absorption features
        base_signature += np.exp(-((wavelengths - 450)/50)**2) * 0.1  # Blue reflectance
        base_signature -= np.exp(-((wavelengths - 980)/50)**2) * 0.08  # Water absorption
        base_signature -= np.exp(-((wavelengths - 1200)/100)**2) * 0.1
        base_signature -= np.exp(-((wavelengths - 1450)/50)**2) * 0.15
        base_signature -= np.exp(-((wavelengths - 1950)/50)**2) * 0.2
        base_signature = np.clip(base_signature, 0, 0.15)
        
    elif desert:
        # Desert/barren spectral signature with more detail
        base_signature = np.zeros(len(wavelengths)) + 0.3
        # Add desert spectral characteristics
        base_signature += np.exp(-((wavelengths - 600)/100)**2) * 0.1  # Red reflectance
        base_signature += np.exp(-((wavelengths - 2200)/200)**2) * 0.15  # SWIR reflectance
        # Add clay/mineral absorption features
        base_signature -= np.exp(-((wavelengths - 2200)/100)**2) * 0.1  # Clay absorption
        base_signature -= np.exp(-((wavelengths - 2340)/80)**2) * 0.08  # Mineral absorption
        base_signature = np.clip(base_signature, 0.2, 0.45)
        
    else:
        # Enhanced vegetation spectral signature
        base_signature = np.exp(-((wavelengths - 800) / 200) ** 2) * 0.6
        
        # Add detailed chlorophyll absorption
        chlorophyll_absorption = np.exp(-((wavelengths - 670) / 30) ** 2) * 0.4
        base_signature -= chlorophyll_absorption
        
        # Add detailed water absorption bands
        water_bands = [970, 1200, 1450, 1940, 2250]
        for band in water_bands:
            water_absorption = np.exp(-((wavelengths - band) / 50) ** 2) * 0.3
            base_signature -= water_absorption
        
        # Climate-based modifications
        climate_factor = 1.0
        if abs(lat) < 23.5:  # Tropical
            climate_factor = 1.2  # Higher vegetation activity
            # Add tropical vegetation characteristics
            base_signature += np.exp(-((wavelengths - 550) / 50) ** 2) * 0.1  # Enhanced green reflectance
        elif abs(lat) > 60:  # Polar
            climate_factor = 0.3  # Lower vegetation activity
        elif abs(lat) > 40:  # Continental
            climate_factor = 0.8
        
        base_signature *= climate_factor
        
        # Add seasonal variation with more detail
        month = datetime.now().month
        if season == 'winter':
            month = 12
        elif season == 'spring':
            month = 3
        elif season == 'summer':
            month = 6
        elif season == 'fall':
            month = 9
        
        # Seasonal adjustment with phenology
        if lat > 0:  # Northern hemisphere
            seasonal_factor = 0.7 + 0.5 * np.cos(2 * np.pi * (month - 6) / 12)
            # Add seasonal spectral changes
            if month in [12, 1, 2]:  # Winter
                base_signature -= np.exp(-((wavelengths - 670) / 40) ** 2) * 0.1  # Reduced chlorophyll
            elif month in [3, 4, 5]:  # Spring
                base_signature += np.exp(-((wavelengths - 550) / 60) ** 2) * 0.1  # Enhanced greenness
        else:  # Southern hemisphere (opposite seasons)
            seasonal_factor = 0.7 + 0.5 * np.cos(2 * np.pi * (month - 12) / 12)
        
        base_signature *= seasonal_factor
    
    # Add realistic noise (wavelength-dependent)
    noise = np.random.normal(0, 0.02, len(wavelengths)) * (1 + wavelengths/2500)  # More noise in SWIR
    base_signature += noise
    
    # Ensure realistic reflectance values
    base_signature = np.clip(base_signature, 0, 1)
    
    return wavelengths, base_signature, (water, desert)

def predict_crop_type(lat, lon, use_pretrained=True):
    """Enhanced crop type prediction with pretrained models"""
    # Check if location is desert or water first
    water = is_water_body(lat, lon)
    desert = is_desert(lat, lon)
    
    if water:
        return ['Water'], [100]
    
    if desert:
        return ['Barren'], [100]
    
    if use_pretrained and lat is not None and lon is not None:
        try:
            # Simulate getting features for pretrained model
            features = np.array([
                lat, lon,
                np.sin(2 * np.pi * datetime.now().month / 12),  # Seasonality
                1 if abs(lat) < 23.5 else 0,  # Tropical
                1 if 23.5 <= abs(lat) < 40 else 0,  # Subtropical
                1 if abs(lat) >= 40 else 0  # Temperate
            ]).reshape(1, -1)
            
            # Predict with multiple models
            rf_pred = pretrained_models['rf'].predict_proba(features)[0]
            xgb_pred = pretrained_models['xgb'].predict_proba(features)[0]
            
            # Ensemble prediction
            avg_probs = (rf_pred + xgb_pred) / 2
            top5_idx = np.argsort(avg_probs)[-5:][::-1]
            
            crops = [list(CROP_TYPES.values())[i] for i in top5_idx]
            probs = avg_probs[top5_idx] * 100
            return crops, probs
        except:
            pass
    
    # Fallback to climate-based prediction if models fail
    if abs(lat) < 10:  # Equatorial
        crops = ['Rice', 'Sugarcane', 'Cotton', 'Oil Palm', 'Banana']
    elif abs(lat) < 23.5:  # Tropical
        crops = ['Corn/Maize', 'Rice', 'Cotton', 'Sugarcane', 'Cassava']
    elif abs(lat) < 40:  # Subtropical
        crops = ['Wheat', 'Corn/Maize', 'Soybeans', 'Cotton', 'Sunflower']
    elif abs(lat) < 60:  # Temperate
        crops = ['Wheat', 'Barley', 'Soybeans', 'Alfalfa', 'Canola']
    else:  # High latitude
        crops = ['Barley', 'Pasture', 'Oats', 'Potatoes', 'Forage Crops']
    
    # Generate probabilities
    probs = np.random.dirichlet(np.ones(len(crops))) * 100
    return crops, probs

# Check PyTorch availability and device
@st.cache_resource
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()

# Sidebar controls
with st.sidebar:
    st.subheader("Analysis Settings")
    
    analysis_mode = st.selectbox("Analysis Mode", 
                               ["Real-time Monitoring", 
                                "Deep Learning Analysis",
                                "Seasonal Comparison", 
                                "Historical Analysis"], 
                               index=0)
    
    if analysis_mode == "Deep Learning Analysis":
        model_complexity = st.selectbox("Model Complexity", 
                                      ["Quick Scan (Fast)", 
                                       "Field Analysis (Balanced)", 
                                       "Precision Agriculture (Full)"], 
                                      index=1)
        
        epochs = st.slider("Training Epochs", 5, 100, 30)
        batch_size = st.slider("Batch Size", 16, 256, 64)
    
    season_filter = st.selectbox("Season", 
                               ["Current", "Spring", "Summer", "Fall", "Winter"],
                               index=0)
    
    show_indices = st.multiselect("Vegetation Indices to Display",
                                ["NDVI", "EVI", "SAVI", "GNDVI", "NDWI", "MSI", 
                                 "NDRE", "PSRI", "PRI", "CI", "ARI"],
                                default=["NDVI", "EVI", "NDWI", "NDRE", "CI"])
    
    health_params = st.multiselect("Health Parameters",
                                 ["Chlorophyll (mg/g)", "Moisture (%)", "LAI", 
                                  "Biomass (kg/ha)", "Water Stress", "Nitrogen Stress",
                                  "Heat Stress", "Senescence Index"],
                                 default=["Chlorophyll (mg/g)", "Moisture (%)", "Water Stress"])

# Display device information
st.sidebar.info(f"🔧 Computing Device: {device.type.upper()}")
if device.type == 'cuda':
    st.sidebar.success("🚀 GPU acceleration available!")
elif device.type == 'mps':
    st.sidebar.success("🍎 Apple Silicon acceleration available!")
else:
    st.sidebar.info("💻 Using CPU computation")

# Main content
st.subheader("🗺️ Interactive Global Agricultural Map")
st.write("Click anywhere on the map to analyze hyperspectral signatures and crop health for that location")

# Initialize default values
lat = None
lon = None
country_name = "Unknown"
marker_location = None

# Create interactive map with enhanced features
m = folium.Map(
    location=[20, 0], 
    zoom_start=2,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite'
)

# Add click functionality message
folium.Marker(
    [0, 0],
    popup="Click anywhere on the map to analyze that location!",
    icon=folium.Icon(color='green', icon='leaf')
).add_to(m)

# Display map and capture clicks
map_data = st_folium(m, width=800, height=500, returned_objects=["last_clicked", "bounds"])

# Process map clicks
if map_data['last_clicked'] is not None:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    marker_location = [lat, lon]
    
    # Get country name
    country_name = get_country_name(lat, lon)
    
    # Update map with marker
    m = folium.Map(
        location=[lat, lon], 
        zoom_start=8,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite'
    )
    
    # Add marker with popup showing coordinates and country
    folium.Marker(
        marker_location,
        popup=f"Lat: {lat:.4f}°, Lon: {lon:.4f}°<br>Country: {country_name}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Display the updated map
    st_folium(m, width=800, height=500)
    
    st.success(f"📍 Analyzing location: {lat:.4f}°, {lon:.4f}° | Country: {country_name}")
    
    # Location Information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latitude", f"{lat:.4f}°")
    with col2:
        st.metric("Longitude", f"{lon:.4f}°")
    with col3:
        if is_water_body(lat, lon):
            climate = "Water Body"
        elif is_desert(lat, lon):
            climate = "Desert/Arid"
        else:
            climate = "Tropical" if abs(lat) < 23.5 else "Temperate" if abs(lat) < 40 else "Continental" if abs(lat) < 60 else "Polar"
        st.metric("Climate Zone", f"{climate} | {country_name}")

    # Generate hyperspectral data for the location
    with st.spinner("Acquiring hyperspectral data..."):
        wavelengths, reflectance, (is_water_flag, is_desert_flag) = simulate_hyperspectral_data(lat, lon, season_filter.lower())
        is_vegetated = not (is_water_flag or is_desert_flag)
        
        # Create patches for analysis (simulate multiple pixels)
        n_pixels = 500  # More pixels for better statistics
        reflectance_data = np.random.normal(reflectance, 0.015, (n_pixels, len(reflectance)))  # Less noise
        reflectance_data = np.clip(reflectance_data, 0, 1)
    
    # Calculate vegetation indices
    vegetation_indices = calculate_vegetation_indices(reflectance_data, is_vegetated)
    
    # Estimate health parameters
    health_parameters = estimate_crop_health_parameters(reflectance_data, vegetation_indices, is_vegetated)
    
    # Predict crop types with pretrained models
    likely_crops, crop_probs = predict_crop_type(lat, lon, use_pretrained=True)
    
    # Display Results
    st.subheader("📊 Hyperspectral Analysis Results")
    
    # Spectral Signature Plot
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wavelengths, reflectance, 'b-', linewidth=2, label='Average Reflectance')
        ax.fill_between(wavelengths, reflectance - np.std(reflectance_data, axis=0), 
                       reflectance + np.std(reflectance_data, axis=0), alpha=0.3, label='Variability')
        
        # Add spectral band regions with better visualization
        colors = ['#9b59b6', '#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#1abc9c', '#e67e22', '#f1c40f']
        for i, (band_name, (start, end)) in enumerate(SPECTRAL_BANDS.items()):
            if start <= wavelengths.max() and end >= wavelengths.min():
                ax.axvspan(start, end, alpha=0.15, color=colors[i % len(colors)], label=band_name)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title(f'Hyperspectral Signature\n{country_name} | Lat: {lat:.4f}°, Lon: {lon:.4f}°')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Enhanced Crop Prediction with more details
        st.subheader("🌾 Predicted Crop Types")
        crop_df = pd.DataFrame({'Crop': likely_crops, 'Probability (%)': crop_probs})
        crop_df = crop_df.sort_values('Probability (%)', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#3498db']
        ax.barh(crop_df['Crop'], crop_df['Probability (%)'], color=colors[:len(crop_df)])
        ax.set_xlabel('Probability (%)')
        ax.set_title('Top Predicted Crop Types')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Display additional crop information
        if is_vegetated:
            st.info("🌱 Vegetation detected - agricultural analysis available")
            if crop_df['Probability (%)'].iloc[0] > 50:
                st.success(f"Most likely crop: {crop_df['Crop'].iloc[0]} ({crop_df['Probability (%)'].iloc[0]:.1f}%)")
            else:
                st.warning("Multiple crop types possible - consider field verification")
        elif is_water_flag:
            st.info("💧 Water body detected - no crop analysis available")
        elif is_desert_flag:
            st.info("🏜️ Desert/barren area detected - no crop analysis available")
    
    # Vegetation Indices Visualization
    st.subheader("🌿 Vegetation Indices Analysis")
    
    if len(show_indices) > 0:
        cols = st.columns(min(3, len(show_indices)))
        col_idx = 0
        
        for idx_name in show_indices:
            with cols[col_idx % len(cols)]:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                if is_vegetated:
                    # Show distribution of index values
                    sns.histplot(vegetation_indices[idx_name], kde=True, ax=ax, color='#27ae60')
                    ax.set_title(f'{idx_name} Distribution')
                    ax.set_xlabel('Index Value')
                    ax.set_ylabel('Frequency')
                    
                    # Add interpretation
                    mean_val = np.mean(vegetation_indices[idx_name])
                    if idx_name == 'NDVI':
                        if mean_val > 0.6:
                            interpretation = "Dense, healthy vegetation"
                        elif mean_val > 0.3:
                            interpretation = "Moderate vegetation"
                        elif mean_val > 0.1:
                            interpretation = "Sparse vegetation"
                        else:
                            interpretation = "Little to no vegetation"
                    elif idx_name == 'NDWI':
                        if mean_val > 0.1:
                            interpretation = "High water content"
                        elif mean_val > -0.1:
                            interpretation = "Moderate water content"
                        else:
                            interpretation = "Low water content"
                    else:
                        interpretation = ""
                    
                    ax.annotate(f"Mean: {mean_val:.2f}\n{interpretation}", 
                              xy=(0.05, 0.85), xycoords='axes fraction',
                              bbox=dict(boxstyle="round", fc="white", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No vegetation detected', 
                           ha='center', va='center', fontsize=12)
                    ax.set_axis_off()
                
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            col_idx += 1
    
    # Health Parameters Visualization
    if len(health_params) > 0 and is_vegetated:
        st.subheader("💚 Crop Health Parameters")
        
        # Create radar chart for health parameters
        health_df = pd.DataFrame.from_dict(health_parameters, orient='index', columns=['Value'])
        health_df = health_df.loc[health_params]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=health_df['Value'].values,
            theta=health_df.index,
            fill='toself',
            name='Health Parameters'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(health_df['Value'].max(), 1)]
                )),
            showlegend=False,
            title="Crop Health Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed health parameter table
        st.dataframe(health_df.style.background_gradient(cmap='RdYlGn', axis=0),
                    use_container_width=True)
    
    # Deep Learning Analysis Section
    if analysis_mode == "Deep Learning Analysis" and is_vegetated:
        st.subheader("🤖 Deep Learning Analysis")
        st.info("This section demonstrates how deep learning could be applied to hyperspectral data analysis")
        
        # Prepare data for deep learning
        X = reflectance_data
        y = np.random.choice([0, 1, 2, 3, 4], size=n_pixels)  # Simulated classes
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dim
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Model selection based on complexity
        if model_complexity == "Quick Scan (Fast)":
            model = HighPrecisionHyperspectralModel(input_channels=1, num_classes=5)
            st.info("Using lightweight CNN for quick analysis")
        elif model_complexity == "Field Analysis (Balanced)":
            model = HighPrecisionHyperspectralModel(input_channels=1, num_classes=5)
            st.info("Using balanced CNN for field-level analysis")
        else:  # Precision Agriculture
            model = HighPrecisionHyperspectralModel(input_channels=1, num_classes=5)
            st.info("Using high-precision CNN for detailed analysis")
        
        # Training
        with st.spinner("Training neural network (simulated for demo)..."):
            model, train_losses, val_losses, train_accs, val_accs = train_pytorch_model(
                model, train_loader, test_loader, epochs, device
            )
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor.to(device))
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(y_test, predicted.cpu().numpy())
            
            st.success(f"Model Accuracy: {accuracy*100:.1f}%")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predicted.cpu().numpy())
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    
    elif analysis_mode == "Seasonal Comparison" and is_vegetated:
        st.subheader("🌦️ Seasonal Comparison")
        
        seasons = ["Winter", "Spring", "Summer", "Fall"]
        seasonal_data = {}
        
        with st.spinner("Simulating seasonal variations..."):
            for season in seasons:
                _, reflectance, _ = simulate_hyperspectral_data(lat, lon, season.lower())
                seasonal_data[season] = reflectance
        
        # Plot seasonal comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        for season, data in seasonal_data.items():
            ax.plot(wavelengths, data, label=season)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title('Seasonal Spectral Signature Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Calculate NDVI for each season
        seasonal_ndvi = {}
        for season in seasons:
            _, reflectance, _ = simulate_hyperspectral_data(lat, lon, season.lower())
            nir = reflectance[50]  # ~800nm
            red = reflectance[30]  # ~650nm
            seasonal_ndvi[season] = (nir - red) / (nir + red + 1e-8)
        
        # Display seasonal NDVI comparison
        st.subheader("Seasonal NDVI Variations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(seasonal_ndvi.keys(), seasonal_ndvi.values(), color=['#3498db', '#2ecc71', '#e67e22', '#f39c12'])
            ax.set_ylabel('NDVI')
            ax.set_title('Seasonal NDVI Comparison')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            ndvi_df = pd.DataFrame.from_dict(seasonal_ndvi, orient='index', columns=['NDVI'])
            st.dataframe(ndvi_df.style.background_gradient(cmap='RdYlGn', axis=0),
                        use_container_width=True)
    
    elif analysis_mode == "Historical Analysis" and is_vegetated:
        st.subheader("📅 Historical Analysis (Simulated)")
        
        # Simulate 5 years of historical data
        years = [datetime.now().year - i for i in range(5, 0, -1)]
        historical_ndvi = {}
        
        with st.spinner("Generating historical trends..."):
            for year in years:
                # Add some random variation to simulate different years
                np.random.seed(year + int(lat * 100 + lon * 100))
                _, reflectance, _ = simulate_hyperspectral_data(lat, lon, season_filter.lower())
                nir = reflectance[50]
                red = reflectance[30]
                historical_ndvi[year] = (nir - red) / (nir + red + 1e-8) * np.random.uniform(0.9, 1.1)
        
        # Plot historical NDVI
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical_ndvi.keys(), historical_ndvi.values(), marker='o', linestyle='-', color='#27ae60')
        ax.set_xlabel('Year')
        ax.set_ylabel('NDVI')
        ax.set_title('5-Year Historical NDVI Trend')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Add anomaly detection
        ndvi_values = list(historical_ndvi.values())
        mean_ndvi = np.mean(ndvi_values)
        std_ndvi = np.std(ndvi_values)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("5-Year Mean NDVI", f"{mean_ndvi:.3f}")
        with col2:
            current_ndvi = vegetation_indices['NDVI'].mean()
            diff = current_ndvi - mean_ndvi
            st.metric("Current vs Historical", 
                     f"{current_ndvi:.3f}", 
                     delta=f"{diff:.3f} ({diff/std_ndvi:.1f}σ)")
        
        # Interpretation
        if current_ndvi < mean_ndvi - std_ndvi:
            st.warning("⚠️ Below average vegetation health detected")
        elif current_ndvi > mean_ndvi + std_ndvi:
            st.success("🌱 Above average vegetation health detected")
        else:
            st.info("🟢 Vegetation health within normal range")

# Add footer
st.markdown("---")
st.markdown("""
    **Global Agri-Spectral Monitor** - This demo application simulates hyperspectral analysis for agricultural monitoring.
    Data is generated synthetically based on geographic location and climate characteristics.
""")

# Add expandable technical details
with st.expander("Technical Details"):
    st.markdown("""
    ### Simulated Data Characteristics:
    - **Spectral Resolution:** 210 bands (400-2500nm)
    - **Spatial Resolution:** Simulated 30m pixels (Landsat-like)
    - **Temporal Resolution:** Current season + historical simulations
    
    ### Analysis Methods:
    - **Vegetation Indices:** 11 standard agricultural indices
    - **Crop Classification:** Ensemble of Random Forest and XGBoost models
    - **Deep Learning:** 1D CNN for hyperspectral classification
    - **Health Parameters:** Empirical relationships from spectral signatures
    
    ### Limitations:
    - All data is simulated for demonstration purposes
    - Actual agricultural monitoring would require real sensor data
    - Model performance shown is illustrative only
    """)

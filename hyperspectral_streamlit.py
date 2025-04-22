import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv3D, MaxPooling3D, Flatten, Dense, 
                                   Dropout, BatchNormalization, GlobalAveragePooling3D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import time
import pandas as pd
from PIL import Image

# Set up Streamlit
st.set_page_config(page_title="Agri-Spectral Analyst", layout="wide")
st.title("🌱 Agri-Spectral Analyst: Crop Health Monitoring")

# Add agricultural branding
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=100)
st.sidebar.header("Farm Configuration")

# Crop type mapping (for real-world context)
CROP_TYPES = {
    1: "Corn", 2: "Soybeans", 3: "Wheat", 4: "Alfalfa",
    5: "Potatoes", 6: "Tomatoes", 7: "Grapes", 8: "Orchard",
    9: "Pasture", 10: "Woods", 11: "Buildings"
}

# Sidebar controls
with st.sidebar:
    model_type = st.selectbox("Model Intensity", 
                            ["Quick Scan (Fast)", 
                             "Field Analysis (Balanced)", 
                             "Precision Agriculture (Full)"], 
                            index=1)
    
    patch_size = st.slider("Analysis Window Size", 5, 15, 9, step=2,
                          help="Larger windows capture more spatial context but slow processing")
    epochs = st.slider("Training Passes", 5, 30, 10,
                      help="More passes can improve accuracy but increase time")
    batch_size = st.slider("Batch Size", 16, 128, 32,
                         help="Larger batches speed up training but need more memory")

# File upload with agricultural context
col1, col2 = st.columns(2)
with col1:
    data_file = st.file_uploader("Upload Field Spectral Data (.npy)", 
                               type="npy",
                               help="Hyperspectral cube of your agricultural field")
with col2:
    label_file = st.file_uploader("Upload Crop Labels (.npy)", 
                                type="npy",
                                help="Ground truth crop type map")

# Optimized patch extraction for agricultural data
def extract_field_patches(data, labels, patch_size=9):
    margin = patch_size // 2
    padded_data = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    labeled_coords = np.argwhere(labels > 0)
    
    # Pre-allocate arrays
    patches = np.zeros((len(labeled_coords), patch_size, patch_size, data.shape[2]), dtype=np.float32)
    patch_labels = np.zeros(len(labeled_coords), dtype=np.int32)
    
    for idx, (i, j) in enumerate(labeled_coords):
        patches[idx] = padded_data[i:i+patch_size, j:j+patch_size, :]
        patch_labels[idx] = labels[i, j] - 1
        
    return patches, patch_labels

# Agricultural-focused model architectures
def build_quick_scan_model(input_shape, n_classes):
    model = Sequential([
        Conv3D(8, (3, 3, 3), activation='relu', input_shape=input_shape),
        MaxPooling3D((2, 2, 2)),
        BatchNormalization(),
        GlobalAveragePooling3D(),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    return model

def build_field_analysis_model(input_shape, n_classes):
    model = Sequential([
        Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D((2, 2, 1)),
        
        Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D((2, 2, 1)),
        
        GlobalAveragePooling3D(),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_precision_ag_model(input_shape, n_classes):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling3D((2, 2, 1)),
        
        Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D((2, 2, 1)),
        
        Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling3D(),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    return model

if data_file and label_file:
    # Load data with agricultural context
    with st.spinner("Processing field data..."):
        data = np.load(data_file)
        labels = np.load(label_file)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Field Overview Dashboard
    st.subheader("🌾 Field Overview Dashboard")
    
    # Create agricultural metrics
    total_pixels = labels.size
    labeled_pixels = np.sum(labels > 0)
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    crop_distribution = {CROP_TYPES.get(i, f"Class {i}"): count 
                       for i, count in zip(unique_labels, counts)}
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Field Size (pixels)", total_pixels)
    col2.metric("Crop Coverage", f"{labeled_pixels/total_pixels:.1%}")
    col3.metric("Crop Varieties", len(crop_distribution))
    
    # Crop distribution pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(crop_distribution.values(), labels=crop_distribution.keys(),
          autopct='%1.1f%%', startangle=90)
    ax.set_title("Crop Distribution")
    st.pyplot(fig)
    
    # PCA Visualization for key agricultural bands
    with st.spinner("Identifying key spectral signatures..."):
        pca_result = PCA(n_components=3).fit_transform(data.reshape(-1, data.shape[-1]))
        pca_img = pca_result.reshape(data.shape[0], data.shape[1], 3)
    
    # Field Visualization
    st.subheader("🔍 Field Visualization")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow((pca_img - pca_img.min()) / (pca_img.max() - pca_img.min()))
    ax1.set_title("Spectral Signature Map")
    ax2.imshow(labels, cmap='tab20')
    ax2.set_title("Crop Type Map")
    st.pyplot(fig)
    
    # Model training section
    st.subheader("🚜 Training Crop Analysis Model")
    
    with st.spinner("Preparing field samples..."):
        X, y = extract_field_patches(data, labels, patch_size)
        X = X[..., np.newaxis]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)
    
    # Model selection
    model_builders = {
        "Quick Scan (Fast)": build_quick_scan_model,
        "Field Analysis (Balanced)": build_field_analysis_model,
        "Precision Agriculture (Full)": build_precision_ag_model
    }
    
    # Training with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    training_placeholder = st.empty()
    
    class TrainingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.3f} | Accuracy: {logs['accuracy']:.2f}")
            
            # Update training plot every epoch
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(range(epoch+1), self.model.history.history['loss'], label='Training')
            if 'val_loss' in self.model.history.history:
                ax[0].plot(range(epoch+1), self.model.history.history['val_loss'], label='Validation')
            ax[0].set_title('Training Progress')
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()
            
            ax[1].plot(range(epoch+1), self.model.history.history['accuracy'], label='Training')
            if 'val_accuracy' in self.model.history.history:
                ax[1].plot(range(epoch+1), self.model.history.history['val_accuracy'], label='Validation')
            ax[1].set_title('Accuracy Progress')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Accuracy')
            ax[1].legend()
            
            training_placeholder.pyplot(fig)
            plt.close()
    
    # Build and train model
    start_time = time.time()
    model = model_builders[model_type](X_train.shape[1:], y_train_cat.shape[1])
    
    history = model.fit(
        X_train, y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[
            TrainingCallback(),
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5)
        ],
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Results
    st.success(f"Model trained in {training_time:.1f} seconds")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Agricultural Results Dashboard
    st.subheader("📊 Field Analysis Report")
    
    # Accuracy by crop type
    results_df = pd.DataFrame({
        'Crop': [CROP_TYPES.get(i+1, f"Class {i+1}") for i in y_test],
        'Actual': [CROP_TYPES.get(i+1, f"Class {i+1}") for i in y_test],
        'Predicted': [CROP_TYPES.get(i+1, f"Class {i+1}") for i in y_pred],
        'Correct': y_test == y_pred
    })
    
    accuracy_by_crop = results_df.groupby('Crop')['Correct'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy_by_crop.plot(kind='barh', ax=ax, color='green')
    ax.set_title("Accuracy by Crop Type")
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    st.pyplot(fig)
    
    # Field Health Map (sample prediction visualization)
    st.subheader("🌱 Field Health Map")
    
    # Sample a small portion of the field for visualization
    sample_size = min(100, len(X_test))
    sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
    sample_preds = y_pred[sample_idx]
    
    # Create a mini-map
    map_size = int(np.sqrt(sample_size)) + 1
    pred_map = np.zeros((map_size, map_size), dtype=int)
    true_map = np.zeros((map_size, map_size), dtype=int)
    
    for idx, (i, j) in enumerate([(x//map_size, x%map_size) for x in range(sample_size)]):
        pred_map[i, j] = sample_preds[idx] + 1
        true_map[i, j] = y_test[sample_idx][idx] + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(true_map, cmap='tab20')
    ax1.set_title("Actual Crops")
    ax2.imshow(pred_map, cmap='tab20')
    ax2.set_title("Predicted Crops")
    st.pyplot(fig)
    
    # Agricultural recommendations
    st.subheader("🧑‍🌾 Field Management Recommendations")
    
    # Simple health assessment (for demo purposes)
    avg_confidence = np.max(model.predict(X_test), axis=1).mean()
    
    if avg_confidence > 0.85:
        st.success("✅ Strong crop identification confidence - field appears healthy")
        st.write("Recommended actions:")
        st.write("- Continue current management practices")
        st.write("- Monitor for seasonal changes")
    elif avg_confidence > 0.7:
        st.warning("⚠️ Moderate identification confidence - check field conditions")
        st.write("Recommended actions:")
        st.write("- Conduct ground verification of uncertain areas")
        st.write("- Consider soil nutrient testing")
    else:
        st.error("❌ Low identification confidence - potential issues detected")
        st.write("Recommended actions:")
        st.write("- Immediate field inspection recommended")
        st.write("- Check for disease, pests, or nutrient deficiencies")
    
    # Save model for future use
    if st.button("💾 Save Field Analysis Model"):
        model.save("crop_analysis_model.h5")
        st.success("Model saved for future field analysis")

else:
    st.info("👩‍🌾 Welcome to Agri-Spectral Analyst! Upload your field data to begin crop health monitoring.")
    st.image("https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
            caption="Upload your hyperspectral field data to get started")
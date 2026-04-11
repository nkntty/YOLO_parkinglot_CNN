
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="centered"
)

st.title("🅿️ Smart Parking Space Classifier")
st.markdown("Upload a parking space image to check if it is **Free** or **Busy**.")

# ── Load model ───────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

model = load_model()

# ── GradCAM ──────────────────────────────────────
def make_gradcam(img_array, model):
    mobilenet = model.get_layer('mobilenetv2_1.00_224')
    grad_model = tf.keras.Model(
        inputs  = mobilenet.inputs,
        outputs = [mobilenet.get_layer('Conv_1').output,
                   mobilenet.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, mobilenet_out = grad_model(img_array)
        x = model.get_layer('global_average_pooling2d')(mobilenet_out)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout')(x)
        predictions = model.get_layer('dense_1')(x)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_array, heatmap, alpha=0.4):
    img = (img_array[0] * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (img * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return overlay

# ── Upload image ─────────────────────────────────
uploaded = st.file_uploader(
    "Choose a parking space image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Predict
    prob = model.predict(img_array, verbose=0)[0][0]
    label = "🚗 Busy" if prob > 0.5 else "✅ Free"
    confidence = prob if prob > 0.5 else 1 - prob

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.markdown(f"### Prediction: {label}")
        st.metric("Confidence", f"{confidence:.1%}")
        if prob > 0.5:
            st.error(f"This space is **occupied** ({prob:.1%} probability)")
        else:
            st.success(f"This space is **free** ({(1-prob):.1%} probability)")

    # GradCAM
    st.markdown("---")
    st.markdown("### 🔍 GradCAM — What the model focuses on")
    heatmap = make_gradcam(img_array, model)
    overlay = overlay_gradcam(img_array, heatmap)

    col3, col4 = st.columns(2)
    with col3:
        st.image(heatmap, caption="Heatmap", use_column_width=True, clamp=True)
    with col4:
        st.image(overlay, caption="Overlay", use_column_width=True)

    st.markdown("---")
    st.caption("Smart Parking AI · MobileNetV2 + GradCAM · Built with TensorFlow & Streamlit")

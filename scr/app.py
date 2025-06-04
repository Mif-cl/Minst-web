import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from Mnist_infer import ImageClassifier, predict_image


import psycopg2

# Load credentials from st.secrets
db_config = st.secrets["database"]

# Connect to Supabase PostgreSQL
conn = psycopg2.connect(
    host=db_config["host"],
    port=db_config["port"],
    database=db_config["database"],
    user=db_config["user"],
    password=db_config["password"]
)

def log_prediction(predicted_digit, true_label):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO digit_predictions (timestamp, predicted_digit, true_label) VALUES (NOW(), %s, %s)",
        (predicted_digit, true_label)
    )
    conn.commit()
    cursor.close()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
@st.cache_resource
def load_model():
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load('checkpoint.pt', map_location=device))
    model.eval()
    return model

model = load_model()

# Add CSS to fix dark mode button visibility
st.markdown("""
<style>
/* Fix canvas buttons for dark mode */
.stButton > button {
    background-color: #0066cc !important;
    color: white !important;
    border: 1px solid #0066cc !important;
}

.stButton > button:hover {
    background-color: #0052a3 !important;
    border-color: #0052a3 !important;
}

/* Ensure canvas toolbar buttons are visible in dark mode */
div[data-testid="stVerticalBlock"] button {
    background-color: #f0f0f0 !important;
    color: #333 !important;
    border: 1px solid #ccc !important;
}

div[data-testid="stVerticalBlock"] button:hover {
    background-color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🖌️ Handwritten Digit Recognizer")

st.subheader("Draw a digit (0–9)")

# Initialize session state for canvas clearing
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

canvas_result = st_canvas(
    fill_color="#666363",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",  # Dynamic key for clearing
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
        img = ImageOps.invert(img).convert("L")
        img.save("temp_digit.png")

        pred_class, confidence, probs, _ = predict_image(model, "temp_digit.png", device)

        st.image(img.resize((140, 140)), caption="Your Drawing", width=140)
        st.markdown(f"### 🔢 Predicted Digit: `{pred_class}`")
        st.markdown(f"### 📈 Confidence: `{confidence:.2%}`")

        with st.expander("View all class probabilities"):
            for i, p in enumerate(probs):
                st.write(f"{i}: {p:.2%}")

        true_label = st.text_input("✍️ Enter the correct digit (for feedback collection):", "")
        if true_label.isdigit() and 0 <= int(true_label) <= 9:
            true_label_int = int(true_label)
            log_prediction(pred_class, true_label_int)
            st.success(f"✅ Logged! You marked the correct digit as: {true_label}")
            
            # Clear the canvas by incrementing the key
            st.session_state.canvas_key += 1
            st.rerun()  # Refresh the app to show cleared canvas
            
        elif true_label:
            st.error("Please enter a digit between 0 and 9.")
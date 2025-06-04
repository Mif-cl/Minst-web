import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from Mnist_infer import ImageClassifier, predict_image


import psycopg2

import os
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

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

#   Session State Initialization  
if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False
if "canvas_rerun_count" not in st.session_state:
    st.session_state.canvas_rerun_count = 0
if "feedback_submitted_successfully" not in st.session_state:
    st.session_state.feedback_submitted_successfully = False

#   App Title and Drawing Canvas  

st.title("🖌️ Handwritten Digit Recognizer")

st.subheader("Draw a digit (0–9)")

# Reset canvas if clear_canvas flag is True
canvas_key = "canvas"
# If we need to clear, we change the key to force a re-render of the canvas
if st.session_state.clear_canvas:
    # Append a unique identifier to the key to force a new canvas instance
    # This is the Streamlit-recommended way to clear a component
    canvas_key = "canvas_" + str(st.session_state.get("canvas_rerun_count", 0) + 1)
    st.session_state.canvas_rerun_count = st.session_state.get("canvas_rerun_count", 0) + 1
    st.session_state.clear_canvas = False # Reset the flag immediately

canvas_result = st_canvas(
    fill_color="#3D3A3A5A",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=canvas_key,
)


#   Predict Button and Logic  
predict_clicked = st.button("Predict", help="Click to predict the drawn digit")

if predict_clicked:
    # Validate if something is drawn
    if canvas_result.image_data is None or canvas_result.image_data[:, :, 0].sum() == 0:
        st.warning("Please draw a digit on the canvas before predicting!")
    else:
        # Process the image and predict
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype("uint8"))
        img = ImageOps.invert(img).convert("L")
        img.save("temp_digit.png") # Save temporarily for prediction

        pred_class, confidence, probs = predict_image(model, "temp_digit.png", device)

        st.image(img.resize((140, 140)), caption="Your Drawing", width=140)
        st.markdown(f"### 🔢 Predicted Digit: `{pred_class}`")
        st.markdown(f"### 📈 Confidence: `{confidence:.2%}`")

        with st.expander("View all class probabilities"):
            for i, p in enumerate(probs):
                st.write(f"{i}: {p:.2%}")

    #   Feedback Form  
    with st.form("feedback_form", clear_on_submit=False): # Keep False to retain input on error
        st.session_state.current_pred_class = pred_class # Store for logging
        true_label_input = st.text_input("✍️ Enter the correct digit (for feedback collection):", key="true_label_input_key")
        submitted_feedback = st.form_submit_button("Submit Feedback")

        if submitted_feedback:
            if true_label_input.isdigit() and 0 <= int(true_label_input) <= 9:
                true_label_int = int(true_label_input)
                
                # Log the prediction with the actual digit
                log_prediction(st.session_state.current_pred_class, true_label_int)
                
                st.success(f"✅ Logged! You marked the correct digit as: {true_label_int}")
                st.session_state.feedback_submitted_successfully = True
                
            else:
                st.error("Please enter a digit between 0 and 9.")

#   Post-submission Rerun Logic  
# This block runs AFTER the form has been processed in the current script execution.
if st.session_state.feedback_submitted_successfully:
    st.session_state.clear_canvas = True
    st.session_state.canvas_rerun_count += 1
    st.session_state.feedback_submitted_successfully = False # Reset the flag
    # This rerun will cause the entire script to execute again,
    # clearing the canvas by changing its key at the top.
    st.experimental_rerun()

#   Custom CSS for Styling  
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #646cff;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 8em;
        font-weight: 600;
        transition: background-color 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #535bf2;
        color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
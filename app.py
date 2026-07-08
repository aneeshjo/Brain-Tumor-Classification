import streamlit as st

from cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline

from cnnClassifier.ui.styles import load_css
from cnnClassifier.ui.header import render_header
from cnnClassifier.ui.sidebar import render_sidebar
from cnnClassifier.ui.prediction_card import render_prediction_card
from cnnClassifier.ui.probability_card import render_probability_card
from cnnClassifier.ui.footer import render_footer
from cnnClassifier.ui.helper import (
    save_uploaded_file,
    delete_temp_file,
    run_prediction
)
from cnnClassifier.utils.model_downloader import (
    download_model_if_needed
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# Load CSS
# ==========================================================

load_css()

with st.spinner("Preparing AI model..."):
    download_model_if_needed()


# ==========================================================
# Sidebar
# ==========================================================

render_sidebar()


# ==========================================================
# Header
# ==========================================================

render_header()


# ==========================================================
# Upload Section
# ==========================================================

uploaded_file = st.file_uploader(
    label="Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================================
# Prediction Section
# ==========================================================

if uploaded_file is not None:

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.subheader("Uploaded MRI Image")

        st.image(
            uploaded_file,
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction")

        if st.button(
            "🔍 Run Prediction",
            use_container_width=True
        ):

            temp_image_path = None

            try:

                # Save Uploaded Image
                temp_image_path = save_uploaded_file(
                    uploaded_file
                )

                # Prediction Spinner
                with st.spinner(
                    "AI is analyzing the MRI image..."
                ):

                    results = run_prediction(
                            temp_image_path
                    )

                # Prediction Card
                render_prediction_card(
                    results
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Probability Card
                render_probability_card(
                    results
                )

            except Exception as e:

                st.error(
                    f"Prediction Failed\n\n{e}"
                )

            finally:

                if temp_image_path:

                    delete_temp_file(
                        temp_image_path
                    )


# ==========================================================
# Footer
# ==========================================================

render_footer()
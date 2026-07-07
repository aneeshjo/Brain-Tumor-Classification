import streamlit as st


def render_prediction_card(results: dict):

    predicted_class = results["predicted_class"]
    confidence = results["confidence_score"]

    st.success("Prediction Completed")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Class",
            predicted_class
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )
import streamlit as st


def render_probability_card(results: dict):

    probabilities = results["class_probabilities"]

    st.subheader("Prediction Probabilities")

    for cls, prob in sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        st.write(f"**{cls}**")

        st.progress(prob / 100)

        st.caption(f"{prob:.2f}%")
import streamlit as st


def render_footer():

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("🧠 Brain Tumor Classification")

    with col2:
        st.caption("TensorFlow • Streamlit")

    with col3:
        st.caption("Developed by Aneesh Jose")

    st.caption(
        "© 2026 End-to-End Deep Learning Project"
    )
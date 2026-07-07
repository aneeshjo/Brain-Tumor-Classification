import streamlit as st


def render_header() -> None:
    """
    Render application header.
    """

    with st.container(border=True):

        st.title("🧠 Brain Tumor Classification")

        st.caption(
            "End-to-End Deep Learning Application using a Custom CNN"
        )

        st.write(
            """
This application classifies Brain MRI images into one of four categories:

- 🟢 Glioma
- 🟡 Meningioma
- 🔵 No Tumor
- 🟣 Pituitary
"""
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Classes", "4")

        with c2:
            st.metric("Model", "CNN")

        with c3:
            st.metric("Framework", "TensorFlow")

        with c4:
            st.metric("Accuracy", "90.5%")
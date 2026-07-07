import streamlit as st


def render_sidebar() -> None:
    """
    Render the application sidebar.
    """

    with st.sidebar:

        st.title("🧠 Brain Tumor Classifier")

        st.markdown("---")

        st.subheader("📖 About Project")

        st.write(
            """
            This application uses a **Custom Convolutional Neural Network (CNN)**
            built with **TensorFlow/Keras** to classify Brain MRI images into
            four tumor categories.

            It demonstrates an end-to-end Deep Learning pipeline including:

            - Data Ingestion
            - Data Validation
            - Data Transformation
            - Model Training
            - Model Evaluation
            - Prediction Pipeline
            - Streamlit Deployment
            """
        )

        st.markdown("---")

        st.subheader("🤖 Model Information")

        st.write("**Architecture:** Custom CNN")
        st.write("**Framework:** TensorFlow 2.x")
        st.write("**Frontend:** Streamlit")
        st.write("**Input Size:** 224 × 224 × 3")
        st.write("**Classes:** 4")
        st.write("**Validation Accuracy:** ~90.5%")

        st.markdown("---")

        st.subheader("🧬 Supported Classes")

        st.markdown(
            """
            - 🟢 Glioma
            - 🟡 Meningioma
            - 🔵 No Tumor
            - 🟣 Pituitary
            """
        )

        st.markdown("---")

        with st.expander("📚 About the Model"):

            st.write(
                """
                The model was developed using a **Custom CNN**
                architecture implemented entirely in TensorFlow.

                The prediction workflow follows a modular architecture:

                **MRI Image**

                ⬇

                **Prediction Pipeline**

                ⬇

                **Preprocessing**

                ⬇

                **CNN Model**

                ⬇

                **Prediction**

                ⬇

                **Confidence Score**
                """
            )

        with st.expander("⚠ Disclaimer"):

            st.warning(
                """
                This application is intended for **educational and portfolio**
                purposes only.

                It should **not** be used as a substitute for professional
                medical diagnosis or clinical decision-making.
                """
            )

        st.markdown("---")

        st.subheader("👨‍💻 Developer")

        st.write("**Aneesh Jose**")
        st.write("Machine Learning Engineer")

        st.caption(
            "End-to-End Deep Learning Project"
        )

        st.markdown("---")

        st.success("✅ Model Ready for Prediction")
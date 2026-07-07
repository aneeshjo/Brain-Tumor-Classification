import os
import tempfile

import streamlit as st

from cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline


def save_uploaded_file(uploaded_file) -> str:
    """
    Save the uploaded image as a temporary file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        str: Temporary file path.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        return temp_file.name


def delete_temp_file(file_path: str) -> None:
    """
    Delete temporary image file.

    Args:
        file_path (str): Path of temporary file.
    """

    if (
        file_path is not None
        and os.path.exists(file_path)
    ):
        os.remove(file_path)


def run_prediction(image_path: str) -> dict:
    """
    Run the prediction pipeline.

    Args:
        image_path (str): Path of uploaded MRI image.

    Returns:
        dict: Prediction results.
    """

    prediction_pipeline = PredictionPipeline()

    results = prediction_pipeline.run_pipeline(
        image_path=image_path
    )

    return results


def validate_uploaded_file(uploaded_file) -> bool:
    """
    Validate uploaded file extension.

    Args:
        uploaded_file: Streamlit UploadedFile.

    Returns:
        bool
    """

    if uploaded_file is None:
        return False

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    return extension in valid_extensions


def show_error(message: str) -> None:
    """
    Display an error message.

    Args:
        message (str): Error message.
    """

    st.error(message)


def show_success(message: str) -> None:
    """
    Display a success message.

    Args:
        message (str): Success message.
    """

    st.success(message)


def show_warning(message: str) -> None:
    """
    Display a warning message.

    Args:
        message (str): Warning message.
    """

    st.warning(message)


def show_info(message: str) -> None:
    """
    Display an information message.

    Args:
        message (str): Information message.
    """

    st.info(message)
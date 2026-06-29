"""
Project Template Generator
--------------------------
This script creates the complete folder and file structure
for the Brain Tumor Classification project.

Run:
    python template.py
"""

from pathlib import Path
import logging

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:"
)

# ==========================================================
# Project Configuration
# ==========================================================

PROJECT_NAME="cnnClassifier"

# ==========================================================
# List of files and folders to create
# ==========================================================

list_of_files=[
    # Configuration
    "config/config.yaml",
    "config/params.yaml",

    # Research & Notebooks
    "research/trials.ipynb",
    "notebook/01_data_exploration.ipynb",
    "notebook/02_model_experiment.ipynb",

    #package
    f"src/{PROJECT_NAME}/__init__.py",

    # Components
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/components/data_ingestion.py",
    f"src/{PROJECT_NAME}/components/data_validation.py",
    f"src/{PROJECT_NAME}/components/data_transformation.py",
    f"src/{PROJECT_NAME}/components/model_trainer.py",
    f"src/{PROJECT_NAME}/components/model_evaluation.py",

     # Configuration Manager
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",

    # Constants
    f"src/{PROJECT_NAME}/constants/__init__.py",

    # Entity
    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/entity/config_entity.py",

    # Pipeline
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/pipeline/training_pipeline.py",
    f"src/{PROJECT_NAME}/pipeline/prediction_pipeline.py",

    # Utilities
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/utils/common.py",

    # Core
    f"src/{PROJECT_NAME}/logger.py",
    f"src/{PROJECT_NAME}/exception.py",

    # Application
    "app.py",
    "main.py",

    # Root Files
    "README.md",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    "Dockerfile",
    "LICENSE",

    # Test Folder
    "tests/__init__.py",

    # Static & Templates
    "static/.gitkeep",
    "templates/.gitkeep",

    # Artifacts
    "artifacts/.gitkeep"

]

# ==========================================================
# Create Folder Structure
# ==========================================================

for filepath in list_of_files:

    filepath=Path(filepath)

    filedir = filepath.parent

    if filedir != Path(""):
        filedir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

        if not filepath.exists():
            filepath.touch()
            logging.info(f"Created file: {filepath}")

        else:
            logging.info(f"File already exists: {filepath}")

from setuptools import find_packages,setup

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> list:
    """
    Returns a list of requirements.
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements

setup(
    name="cnnClassifier",
    version="0.0.1",
    author="Aneesh Jose",
    author_email="your_email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
)

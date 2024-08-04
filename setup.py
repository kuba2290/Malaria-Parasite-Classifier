from setuptools import find_packages, setup
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() 



def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#') and req != '-e .']
    return requirements

setup(
    name="malaria_project",
    version="0.0.1",
    author="Kuba",
    long_description= long_description,
    long_description_content_type= "text/markdown",
    author_email="punkuba@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    package_data={"": ["*.txt", "*.rst"]},
    include_package_data=True,
)

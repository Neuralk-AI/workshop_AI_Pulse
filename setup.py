import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="workshop_AI_Pulse",
    version="0.0.1",
    author="Alexandre Pasquiou,Vladimir Kondratyev",
    author_email="alex@neuralk-ai.com",
    description="Package for representation of numerical data in format of integers or strings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Neuralk-AI/workshop_AI_Pulse",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "ipykernel",
        "matplotlib",
        "numpy",
        "ruff",
        "sentence-transformers",
        "torch",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
)

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="interpret",
    version="0.0.1",
    author="Thomas Tumiel",
    description="Interpreting deep learning models in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ttumiel/deep-interp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

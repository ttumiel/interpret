import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

test_deps = [
    'coverage',
    'pytest',
]
extras = {
    'test': test_deps,
}

setuptools.setup(
    name="interpret-pytorch",
    version="0.2.0",
    author="Thomas Tumiel",
    description="Interpreting deep learning models in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ttumiel/deep-interp",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'Pillow',
        'tqdm',
        'torchvision',
        'torch'
    ],
    tests_require=test_deps,
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
)

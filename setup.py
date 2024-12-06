from setuptools import setup, find_packages

setup(
    name="mnist_model",
    version="1.0.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'pytest>=7.0.0',
        'pillow>=8.0.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="MNIST Model with parameter constraints",
    keywords="mnist, deep learning, pytorch",
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 
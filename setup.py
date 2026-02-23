from setuptools import setup, find_packages

setup(
    name="wheelgrad",
    version="0.1.0",
    author="WheelGrad Contributors",
    description="Wheel algebra for neural network numerical stability — no more NaN propagation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wheelgrad/wheelgrad",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.21"],
    extras_require={
        "torch": ["torch>=1.9"],
        "dev":   ["pytest", "matplotlib"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="wheel-algebra numerical-stability deep-learning nan pytorch jax",
)

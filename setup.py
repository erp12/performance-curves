import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="performance_curves",
    version="0.0.1.dev0",
    author="Edward Pantrdige",
    author_email="erp12@hampshire.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/erp12/performance-curves",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19.0",
    ],
    tests_require=[
        "scikit-learn>=0.23"
        "pytest"
    ]
)

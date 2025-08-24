from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="samsung-network-traffic-detection",
    version="1.0.0",
    author="Subhradip debray",
    author_email="draysubhradip@gmail.com",
    description="Real-time detection of reel/video traffic vs non-reel/video traffic in social networking applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/samsung-traffic-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.10.0",
            "onnxruntime-gpu>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "samsung-capture=src.inference.capture:main",
            "samsung-dashboard=src.inference.dashboard:main",
            "samsung-train=scripts.train_models:main",
            "samsung-evaluate=scripts.evaluate_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
    },
    zip_safe=False,
)

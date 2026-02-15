from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jarvis-but-mine",
    version="0.1.0",
    author="DaMaker1291",
    author_email="your.email@example.com",
    description="A customizable personal assistant inspired by J.A.R.V.I.S.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DaMaker1291/J.A.R.V.I.S-BUT-MINE",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    install_requires=[
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "SpeechRecognition>=3.8.1",
        "pyttsx3>=2.90",
    ],
    entry_points={
        "console_scripts": [
            "jarvis=jarvis.__main__:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

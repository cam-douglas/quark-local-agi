#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="meta_model",
    version="0.1.0",
    description="Meta-Model AI Assistant (CLI + Web)",
    author="You",
    packages=find_packages(exclude=["tests", "logs", "models"]),
    install_requires=[
        "click>=8.2",
        "transformers>=4.54",
        "sentence-transformers>=2.2",
        "fastapi>=0.95",
        "uvicorn[standard]>=0.22",
        "spacy>=3.5",
        "chromadb>=0.3",
        "prometheus-client>=0.16"
    ],
    entry_points={
        "console_scripts": [
            "meta-model=meta_model.cli:cli",
            "meta-model-web=meta_model.web:main",
        ]
    },
    # py2app support so you can do `python setup.py py2app`
    setup_requires=["py2app"],
    options={
        "py2app": {
            "argv_emulation": True,
            "packages": ["meta_model", "transformers", "click", "fastapi", "uvicorn"],
        }
    },
    python_requires=">=3.9",
)


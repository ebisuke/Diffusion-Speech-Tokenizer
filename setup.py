from setuptools import setup, find_packages

setup(
    name="tadicodec",
    version="0.1.0",
    description="TaDiCodec",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        'torch',
        'soundfile',
    ],
    python_requires='>=3.11',
    classifiers=[
    ],
)

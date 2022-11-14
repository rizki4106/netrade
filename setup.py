from setuptools import setup

setup(
    name="netrade",
    version="1.2.2",
    author="Rizki Maulana",
    author_email="rizkimaulana348@gmail.com",
    packages=["netrade"],
    url="https://github.com/rizki4106/netrade",
    license="LICENCE",
    description="AI trading assistant",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    install_requires=[
        "torch",
        "torchmetrics",
        "scikit-image",
        "Pillow",
        "mplfinance",
        "numpy",
    ]
)
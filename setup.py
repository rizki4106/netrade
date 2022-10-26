from setuptools import setup

setup(
    name="netrade",
    version="1.0.0",
    author="Rizki Maulana",
    author_email="rizkimaulana348@gmail.com",
    packages=["netrade", "netrade.test"],
    url="https://github.com/rizki4106/netrade",
    license="LICENCE",
    description="AI trading assistant",
    long_description=open('README.md').read(),
    install_requires=[
        "torch",
        "torchmetrics",
        "scikit-image",
        "Pillow",
    ]
)
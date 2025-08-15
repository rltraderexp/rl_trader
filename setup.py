import os
from setuptools import setup, find_packages

# Function to read the version from the package's __init__.py
def get_version():
    version_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'rl_trading_project',
        '__init__.py'
    )
    with open(version_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'\"")
    return '0.1.0' # Default version

# Function to read requirements from requirements.txt
def get_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='rl_trading_project',
    version=get_version(),
    packages=find_packages(),
    description='A Reinforcement Learning framework for algorithmic trading strategies.',
    author='Your Name', # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    install_requires=get_requirements(),
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Or your chosen license
        'Operating System :: OS Independent',
    ],
)
from setuptools import setup, find_packages

setup(
    name='identifiability-toy-study',
    version='1.0.1',
    description='Research extensions combining MI-identifiability (Méloux), spd (Goodfire-AI), circuit-stability (Alan Sun), and eap-ig-faithfulness (Hannah W.).',
    url='https://github.com/unrulyabstractions/identifiability-toy-study',
    author='Ian Rios-Sialer',
    author_email='ian@unrulyabstractions.com',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        # Core dependencies for mi_identifiability
        'tqdm~=4.66.5',
        'matplotlib~=3.8.3',
        'numpy~=1.26.4',
        'scipy~=1.12.0',
        'pandas~=2.2.1',
        'torch~=2.4.1',
        'networkx~=3.2.1',
        'torchvision~=0.19.1',
        # Additional dependencies for spd module
        'pydantic>=2.0.0',
        'datasets',
        'transformers',
        'wandb',
        'einops',
        'jaxtyping',
        # Additional dependencies for other modules  
        'plotly',
        'scikit-learn',
        'seaborn',
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8',
            'mypy',
        ]
    },
)

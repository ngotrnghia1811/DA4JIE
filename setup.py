from setuptools import setup, find_packages

setup(
    name='da4jie',
    version='1.0.0',
    description='Domain Adaptation for Joint Information Extraction',
    author='Nghia Trung Ngo',
    author_email='nghian@uoregon.edu',
    url='https://github.com/nghiango1/da4jie',
    packages=find_packages(exclude=['agie', '_paper_latex', 'notebooks']),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
        'transformers==3.5.1',
        'nltk>=3.5',
        'lxml>=4.6',
        'trankit>=1.1.0',
        'tqdm>=4.60',
        'easydict>=1.9',
        'packaging>=20.9',
        'numpy>=1.19',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

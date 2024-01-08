from setuptools import setup, find_packages

setup(
    name='rocca',
    version='0.1.0',  
    author='Jeffri Lieca H',  
    author_email='handoyojeffri@gmail.com',  
    description='Library Machine Learning Python yang simple dan lengkap ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/url-projek-anda',  # Ganti dengan URL repositori GitHub Anda
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',  
)

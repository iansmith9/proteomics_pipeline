from setuptools import setup, find_packages

setup(
    name='proteomics_pipeline',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scikit-learn',
        'scipy',
        'mmh3',
        'matplotlib',
        'pyopenms',
        'pyteomics',
        'click',
        'lxml',
        'pyarrow',
        'black',
        'ppx',
        'biosaur2',
        'mokapot',
        'pyascore @ git+https://github.com/iansmith9/pyAscore.git#egg=pyascore'
    ],
    package_data ={        
    'proteomics_pipeline/comet_params_defaults':['*.parmas','*.fasta']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points = {
        'console_scripts': ['proteomics_pipeline=proteomics_pipeline.main:main'],
    }
)

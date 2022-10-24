import os
import platform
import sys
from pathlib import Path

from pkg_resources import parse_version
from setuptools import find_packages, setup


def main():
    version_file = Path(__file__).parent / 'VERSION'
    with open(str(version_file)) as fin:
        version = fin.read().strip()

    install_requires_base = [
        'absl-py',
        'sox>=1.3.3',
        'soundfile>=0.10.2',
        'nltk>=3.3',
        'pandas>=0.23.3'
    ]

    tensorflow_pypi_dep = [
        'tf-models-official',
        'tensorflow_cloud',
        'tensorflow >= 2.8.0'
    ]
    install_requires = install_requires_base + tensorflow_pypi_dep

    # TODO: Check if this is needed
    #horovod_pypi_dep = [
    #    'horovod[tensorflow] >= 0.21.3'
    #]
    #install_requires = install_requires + horovod_pypi_dep
    

    setup(
        name='tf-deep_speech',
        version=version,
        description='Code for Deep Speech 2 based on Google /tensorflow/models/research/deep_speech',
        url='https://github.com/istvanzk/tf-deep_speech',
        author='Deep Speech 2 authors; Google deep_speech authors; Istvan Z. Kovacs',
        license='Apache Software License 2.0',
        # Classifiers help users find your project by categorizing it.
        #
        # For a list of valid classifiers, see https://pypi.org/classifiers/
        classifiers=[
            'Development Status :: 1 - Planning',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Topic :: Multimedia :: Sound/Audio :: Speech',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Operating System :: OS Independent',
        ],
        #package_dir={'': 'trainer'},
        packages=['trainer', 'model', 'data'],
        python_requires='>=3.8, <4',
        install_requires=install_requires,
        # Install additional data files
        data_files=[
            ('.', [
                'VERSION',
                'run_ds2_train_cloud.sh',
            ])
        ],
        # Install additional package data files
        package_data={
            'data':[
                'vocabulary-hu.txt',
            ]
        },
    )

if __name__ == '__main__':
    main()

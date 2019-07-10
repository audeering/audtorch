from setuptools import setup, find_packages

setup(
    name='audtorch',
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'numpy',
        'audiofile',
        'resampy',
        'torch',
        'pandas',
        'tqdm',
        'tabulate',
    ],
    author=('Andreas Triantafyllopoulos, '
            'Stephan Huber, '
            'Johannes Wagner, '
            'Hagen Wierstorf'),
    author_email='atriant@audeering.com',
    description='Deep learning with PyTorch and audio',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    license='MIT License',
    keywords=['audio'],
    url='https://github.com/audeering/audtorch',
    project_urls={
        'Documentation': 'https://audtorch.readthedocs.io',
        'Tracker': 'https://github.com/audeering/audtorch/issues',
    },
    platforms='any',
    tests_require=['pytest'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Multimedia :: Sound/Audio',
    ],
)

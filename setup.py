import setuptools

setuptools.setup(
    name='sageopt',
    version='0.6.0',
    author='Riley John Murray',
    url='https://github.com/rileyjmurray/sageopt',
    author_email='rmurray201693@gmail.com',
    description='Signomial and polynomial optimization via SAGE relaxations',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    python_requires='>=3.6',
    install_requires=["ecos >= 2",
                      "numpy >= 1.14",
                      "scipy >= 1.1",
                      "nose2",
                      'tqdm'],
    test_suite='nose2.collector'
)

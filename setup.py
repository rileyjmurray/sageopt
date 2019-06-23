import setuptools

setuptools.setup(
    name='sageopt',
    version='0.2.0',
    author='Riley John Murray',
    url='https://github.com/rileyjmurray/sageopt',
    author_email='rmurray@caltech.edu',
    description='Signomial and polynomial optimization via SAGE relaxations',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=["ecos >= 2",
                      "numpy >= 1.14",
                      "scipy >= 1.1"]
)

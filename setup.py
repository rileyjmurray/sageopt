import setuptools

setuptools.setup(
    name='sageopt',
    version='0.2.0',
    author='Riley John Murray',
    author_email='rmurray@caltech.edu',
    description='Signomial and polynomial optimization via SAGE relaxations',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Natural Language :: English',
        'License :: Apache Software License',
        'Development Status :: 4 - Production/Stable',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    install_requires=["ecos >= 2",
                      "numpy >= 1.14",
                      "scipy >= 1.1"]
)

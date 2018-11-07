import sys
from setuptools import find_packages, setup


if sys.version_info < (3,5):
    sys.exit("Requires Python 3.5 or higher")

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license_ = f.read()

REQUIRED_PACKAGES = [
    'cython >= 0.28.4',
    'matplotlib >= 2.2.2',
    'numpy >= 1.15.0',
    'pathos >= 0.2.2',
    'Pillow >= 5.2.0',
    'scikit-learn >= 0.19.2',
    'scipy >= 1.1.0',
    'SimpleITK >= 1.1.0',
    'tensorboardX',
    'tensorflow',
    'torch',
]

TEST_PACKAGES = [

]

setup(
    name='MIALab',
    version='0.1.0',
    description='medical image analysis laboratory',
    long_description=readme,
    author='Medical Image Analysis Group',
    author_email='mauricio.reyes@istb.unibe.ch',
    url='https://github.com/istb-mia/MIALab',
    license=license_,
    packages=find_packages(exclude=['test', 'docs']),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries'
    ],
    keywords=[
        'medical image analysis',
        'machine learning',
        'neuro'
    ]
)

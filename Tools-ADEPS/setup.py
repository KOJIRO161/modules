from setuptools import setup, find_packages

setup(
    name='ToolsADEPS',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    author='KOJIRO161',
    author_email='okazaki.y.9885@m.isct.ac.jp',
    description='for Analyzing Data for Earth and Planetary Sciences',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/KOJIRO161/modules/Tools-ADEPS',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.13',
)

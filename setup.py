from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="restaurant-analytics-app",
    version="1.0.0",
    author="Vasim",
    author_email="vasimjafarshaik@gmail.com",
    description="A multi-task Flask application for restaurant analytics including rating prediction, recommendations, cuisine classification, and data visualization",
    long_description=open("README.md").read() if open("README.md", "r").read() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/vasim/restaurant-analytics-app",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: Flask",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "restaurant-analytics=main_app:main",
        ],
    },
    keywords="flask machine-learning restaurant-analytics prediction recommendation classification visualization",
    project_urls={
        "Bug Reports": "https://github.com/vasim/restaurant-analytics-app/issues",
        "Source": "https://github.com/vasim/restaurant-analytics-app",
    },
)

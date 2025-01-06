from setuptools import setup, find_packages

setup(
    name="startup_wellness_analyze",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.112.2",
        "uvicorn==0.30.6",
        "pydantic[email]==2.8.2",
        "firebase-admin==6.5.0",
        "python-multipart==0.0.9",
        "sqlalchemy==2.0.28",
        "psycopg2-binary==2.9.9",
        "alembic==1.13.1",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "scikit-learn==1.5.1",
        "statsmodels==0.14.2",
        "plotly==5.24.0",
        "dash==2.17.1",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "lifelines==0.29.0"
    ],
    python_requires=">=3.9",
)
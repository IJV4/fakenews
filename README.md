# Fake News Detector

Fake news detection using machine learning.

Online version of this project at:
https://fakenews-ijv.streamlit.app/

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

- Install python 3.8.5 or higher.

- Create an environment using venv

Open a terminal and navigate to your project folder.

```
cd myproject
python -m venv .venv
```

A folder named ".venv" will appear in your project. This directory is where your virtual environment and its dependencies are installed.

- Install jupyter notebook.

- Install the following packages:

```
pip install pandas
pip install numpy
pip install sklearn
pip install matplotlib
pip install seaborn
pip install nltk
pip install pickle
```

- Install streamlit

```
pip install streamlit
```

## Usage

Download the data from https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection/data

To run the project, open the jupyter notebook "FakeNewsPredictor" and run the cells.

This will generate a pickle file "model.pkl" and a "vectorizer.pkl" file.

Run the app locally with streamlit:

```
streamlit run app.py
```

Open your browser and go to http://localhost:8501

## License

This project is licensed under the Creative Commons License.

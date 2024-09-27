# Trading-Repositorio
Proyecto 2: Análisis técnico
---

### Team Members: Mario Alberto Roman, Juan Pablo Dominguez, Fernanda Martin del Campo, and Karitsa Alcaraz
The objective of this project is to create an algorithm that optimizes buy and sell signals as effectively as possible for trading using different models and strategies. We tested the code we created with two different assets: AAPL stock and BTC-USD.

---
## Proyect 2: Technical Analysis
The main objective of this project is to optimize an algorithm for efficient trading using technical indicators like RSI, MACD, Bollinger Bands, SMA, and ATR. We optimize buy/sell signals by adjusting the parameters of each indicator, aiming to find the best combination for positive results. The results are detailed in the PDF.To get started with this repository, you will need to have Python installed on your machine. We recommend using Python 3.10 or higher.

To get started with this repository, you will need to have Python installed. We recommend using Python 3.10 or higher.
1.Fork this repository
2. Clone the repo

```python
git clone <repo_url>
cd Technical_Analysis_proyect
```
3. Create a virtual environment

```python
python -m venv venv
source venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
4. Install the required dependencies

```python
pip install -r requirements.txt
```
## Repository structure

---

### data

---
In this folder, you will find everything related to the datasets we used throughout the project to achieve the results obtained in the PDF. Specifically, it includes the CSV files for BTC and AAPL, in both train and test modalities.
### technical_analysis

---


In this directory, you will find the main functions. In this technical_analysis, whose main objective is to optimize all the technical indicators with their different data inputs and save the results in a DataFrame.



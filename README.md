# Ocean Salinity Regression (Bottle Dataset)

This project uses Python to perform linear regression on the [Kaggle Bottle Dataset](https://www.kaggle.com/datasets/) to predict **water temperature(degree Celcius)** based on **salinity**.

## Project Files
- Bottle_Regression.ipynb – Jupyter notebook containing full analysis
- data/bottle.csv – Dataset used
- models/salinity_model.pkl – Saved regression model
- requirements.txt – Required Python libraries


## What is inside:
- Data Cleaning
- Linear regression with scikit-learn 
- R-squared evaluation and model interpretation 
- Beautiful plots with seaborn/matplotlib
- Plotted regression line 

## Tools used:
 Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## Sample Output:
 R-squared Score: 0.255
 Mean Square Error: 13.293
 Intercept: 167.351
 Coefficient: - 4.624

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook Bottle_Regression.ipynb
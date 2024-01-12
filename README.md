# Used Car Price Regression Analysis and visualization with R

This project was an exercise in analyzing a set of Used Car data using a combination of Python and R (mostly R).

The main goal was to make a regression model in R with the provided data and to create beautiful visuals to go along with it. 

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

car_df1= pd.read_csv("car_data/Car details v3.csv")
car_df1.shape
```
After this, we can see we are dealing with a dataset with 8128 rows and 13 columns. 

# Basic Analysis and visualizations using R
With R, I wanted to create beautiful visuals to further our understanding of the dataset we are working with.
```R
#getting dataset into the file
df_1 <- read.csv("Car details v3.csv")

#import packages needed for project
library(tidyverse)
library(stringr)
library(purrr)
library(randomForest)
library(broom)
library(Amelia)
library(GGally)
library(caret)
library(relaimpo)
library(gbm)
```
A lot of packages for this analysis, but I am going to be running multiple regression analyses on the dataset and these packages are all essential.

## Starting with the ***Brand*** names of the cars, I plotted this bar graph.

```R
#Plotting car name to check the distribution
ggplot(data = df_1, aes(x=name, fill = name)) +
  geom_bar() + labs(x='Car Brand') + labs(title = "Bar Graph of Car Brand") +
  theme(axis.text.x = element_text(angle = 90))
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/5214f296-25c4-411d-b52e-c945261baa0a" alt="Image Description" width="600"/>

## Next, this visualization is a bar graph made to understand the Age distribution of the cars within the dataset.

```R
df_1%>%
group_by(year)%>%
count()%>%
ggplot()+geom_col(aes(y=n,x=year, fill=year))+
labs(title = 'Age Distribution of Cars at the time of Selling',
     subtitle= 'Calculated in 2021',
    x= 'Age of Car',
    y='#Cars')
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/b6647336-3673-42bf-b5cb-ae3293850e8e" alt="Image Description" width="600"/>

```R

```
<img src=" " alt="Image Description" width="600"/>


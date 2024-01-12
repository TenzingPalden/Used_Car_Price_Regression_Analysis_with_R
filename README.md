# Used Car Price Regression Analysis and visualization with R

This project was an exercise in analyzing a set of Used Car data using a combination of ***Python*** and ***R*** (mostly ***R***).

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
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/5214f296-25c4-411d-b52e-c945261baa0a" alt="Image Description" width="500"/>

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
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/b6647336-3673-42bf-b5cb-ae3293850e8e" alt="Image Description" width="500"/>

# We continue this analysis by creating more visualizations to understand this dataset better.
```R
# Bar graph of Fuel
ggplot(data = df_1, aes(x=reorder(fuel, fuel, function(x)-length(x)), fill = fuel)) +
  geom_bar() + labs(x='Fuel type') + labs(title = "Count of Fuel Types in Data set")
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/841c7023-a17c-48c2-8608-43a806994f3f" alt="Image Description" width="500"/>

```R
options(warn=-1)
df_1 %>%
ggplot(aes(x=mileage,fill=fuel))+
geom_density(alpha = 0.5)+
labs(title = 'Distribution of Mileage due to Fuel type',
    x= 'Mileage')
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/16f2d56a-5455-41a1-bba2-620646efdd59" alt="Image Description" width="500"/>

```R
#Histogram of Selling Price
ggplot(df_1, aes(x=selling_price)) + 
  geom_histogram(aes(y=..density..), colour="black")+
  geom_density(alpha=.2, fill="red")+
  labs(x='Selling Price ') + labs(title = "Histogram Graph of Selling Price") +
  scale_x_continuous(trans='log10')
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/5c2e7f35-73b5-45c8-a250-dc6dd302c8f6" alt="Image Description" width="500"/>

```R
#Histogram of Km Driven
ggplot(df_1, aes(x=km_driven*0.621371)) + 
  geom_histogram(color="black", fill="red", bins = 100)+
  labs(x='Miles Driven ') + labs(title = "Histogram of Miles Driven in Dataset") +
  scale_x_continuous(trans='log10')
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/40dd408f-852e-4b6c-ad6f-8df5a95ae11f" alt="Image Description" width="500"/>

# After that, I moved onto cleaning the data and preparing it for regression analysis.
In this part, we will be converting string values to integers.This will allow the Regression analysis to be conducted

```R
#removing torque because that is not important to this analysis
df_1 <- subset (df_1, select = -torque)

#changing car names to integers 
df_1$name <- str_replace(df_1$name, 'Maruti', '0')
df_1$name <- str_replace(df_1$name, 'Skoda', '1')
df_1$name <- str_replace(df_1$name, 'Honda', '2')
df_1$name <- str_replace(df_1$name, 'Hyundai', '3')
df_1$name <- str_replace(df_1$name, 'Toyota', '4')
df_1$name <- str_replace(df_1$name, 'Ford', '5')
df_1$name <- str_replace(df_1$name, 'Renault', '6')
df_1$name <- str_replace(df_1$name, 'Mahindra', '7')
df_1$name <- str_replace(df_1$name, 'Tata', '8')
df_1$name <- str_replace(df_1$name, 'Chevrolet', '9')
df_1$name <- str_replace(df_1$name, 'Fiat', '10')
df_1$name <- str_replace(df_1$name, 'Datsun', '11')
df_1$name <- str_replace(df_1$name, 'Jeep', '12')
df_1$name <- str_replace(df_1$name, 'Mercedes-Benz', '13')
df_1$name <- str_replace(df_1$name, 'Mitsubishi', '14')
df_1$name <- str_replace(df_1$name, 'Audi', '15')
df_1$name <- str_replace(df_1$name, 'Volkswagen', '16')
df_1$name <- str_replace(df_1$name, 'BMW', '17')
df_1$name <- str_replace(df_1$name, 'Nissan', '18')
df_1$name <- str_replace(df_1$name, 'Lexus', '19')
df_1$name <- str_replace(df_1$name, 'Jaguar', '20')
df_1$name <- str_replace(df_1$name, 'Land', '21')
df_1$name <- str_replace(df_1$name, 'MG', '22')
df_1$name <- str_replace(df_1$name, 'Volvo', '23')
df_1$name <- str_replace(df_1$name, 'Daewoo', '24')
df_1$name <- str_replace(df_1$name, 'Kia', '25')
df_1$name <- str_replace(df_1$name, 'Force', '26')
df_1$name <- str_replace(df_1$name, 'Ambassador', '27')
df_1$name <- str_replace(df_1$name, 'Ashok', '28')
df_1$name <- str_replace(df_1$name, 'Isuzu', '29')
df_1$name <- str_replace(df_1$name, 'Opel', '30')
df_1$name <- str_replace(df_1$name, 'Peugeot', '31')

#Converting car name from categorical to numerical value

df_1$name <- as.numeric(df_1$name)
table(df_1$name)
```
Then, Removing unit from mileage, converting it to numeric value and replacing the missing values

```R
df_1$mileage <- str_replace(df_1$mileage, 'kmpl', '')
df_1$mileage <- str_replace(df_1$mileage, 'km/kg', '')
df_1$mileage <- as.numeric(df_1$mileage)
df_1$mileage[is.na(df_1$mileage)]<-mean(df_1$mileage,na.rm=TRUE)
```
Removing unit from engine, converting it to numeric value and replacing the missing values
```R
df_1$engine <- str_replace(df_1$engine, 'CC', '')
df_1$engine <- as.numeric(df_1$engine)
df_1$engine[is.na(df_1$engine)]<-mean(df_1$engine,na.rm=TRUE)
```
Converting seats to numeric value and replacing the missing values
```R
df_1$seats <- as.numeric(df_1$seats)
df_1$seats[is.na(df_1$seats)]<-median(df_1$seats,na.rm=TRUE)
```
Removing unit from max_power, converting it to numeric value and replacing the missing values
```R
df_1$max_power <- str_replace(df_1$max_power, 'bhp', '')
df_1$max_power <- as.numeric(df_1$max_power)
df_1$max_power[is.na(df_1$max_power)]<-mean(df_1$max_power,na.rm=TRUE)
```
Converting fuel into integers
```R
df_1$fuel <- str_replace(df_1$fuel, 'Diesel', "0")
df_1$fuel <- str_replace(df_1$fuel, 'Petrol', "1")
df_1$fuel <- str_replace(df_1$fuel, 'CNG', "2")
df_1$fuel <- str_replace(df_1$fuel, 'LPG', "3")
df_1$fuel <- as.numeric(df_1$fuel)
table(df_1$fuel)
```
Transmission to  binary 0 if Manual and 1 if Automatic
```R
df_1$transmission <- str_replace(df_1$transmission, 'Manual', "0")
df_1$transmission <- str_replace(df_1$transmission, 'Automatic', "1")
df_1$transmission <- as.numeric(df_1$transmission)
table(df_1$transmission)
```
Converting owner into integers
```R
df_1$owner <- str_replace(df_1$owner, 'First Owner', "0")
df_1$owner <- str_replace(df_1$owner, 'Second Owner', "1")
df_1$owner <- str_replace(df_1$owner, 'Third Owner', "2")
df_1$owner <- str_replace(df_1$owner, 'Fourth & Above Owner', "3")
df_1$owner <- str_replace(df_1$owner, 'Test Drive Car', "4")
df_1$owner <- as.numeric(df_1$owner)
table(df_1$owner)
```
Converting seller_type into Ordinal Encoder
```R
df_1$seller_type <- str_replace(df_1$seller_type, "Trustmark Dealer", "0")
df_1$seller_type <- str_replace(df_1$seller_type, "Dealer", "1")
df_1$seller_type <- str_replace(df_1$seller_type, "Individual", "2")
df_1$seller_type <- as.numeric(df_1$seller_type)
table(df_1$seller_type)
```
# Regression Visualizations
I first made a correlation matrix to measure the strength of the relationship between the given variables. 

We can see that selling price is highly correlated to max_power then transmission and name.
```R
library(corrplot)
corrplot(cor(df_1), type="full", 
         method ="color", title = "Correlation Plot", 
         mar=c(0,0,1,0), tl.cex= 0.8, outline= T, tl.col="black")
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/908df0bb-7f6f-4ad6-8d4c-c1b1261d855c" alt="Image Description" width="500"/>

# This will mark the Start of the regression part of the project
Making training and testing splits
```R
set.seed(100)
subset<-sample(nrow(df_1),nrow(df_1)*0.8)
trainSet<-df_1[subset,]
testSet<-df_1[-subset,]
```
linear regression
```R
first_lr <- lm(selling_price ~ ., data = trainSet)
summary(first_lr)
```
eliminate all features with P values above 0.05

```R
main_lr <- lm(selling_price ~ name+ year+ km_driven+ seller_type +transmission + mileage + max_power, data = trainSet)
summary(main_lr)
```
Resulting the equation y=-68240000 + 25190x + 33510x+ -0.8818x +  -116400x + 430700x+ 18740x+ 12440x

I then wanted to see how the test results would fair againt the real results.
```R
pred_lr <- predict(main_lr, newdata = testSet)
error_lr <- testSet$selling_price - pred_lr
RMSE_lr <- round(sqrt(mean(error_lr^2)),2)
RMSE_lr
```

```R
plot(testSet$selling_price,pred_lr, main="Scatterplot", col = c("green","blue"), xlab = "Actual selling_price", ylab = "Predicted selling_price")
```
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/3ad4b12c-2ec0-41df-9344-1ece8fea0aa2" alt="Image Description" width="500"/>

# Afther the linear regression, I moved onto random forest using the rf package in R.
This will mark the Start of the Random Forest part of the project. This was done to understand which feature was most important in deperminting selling price of the car.


```R
rf <- randomForest(selling_price~.,data = trainSet)
plot(rf)
```

<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/6da4b52d-22da-4a19-9018-0be74a13f704" alt="Image Description" width="500"/>
<img src="https://github.com/TenzingPalden/used_car_price_regression_analysis/assets/85039775/dcafc242-f60d-430e-a60b-bf8e1fef6d53" alt="Image Description" width="500"/>




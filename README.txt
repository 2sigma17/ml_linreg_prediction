Objective:
Using historical daily prices of btc/usd and defining exploratory variables to create a predication model for btc/usd. 

Steps:
1. Retrieve historical daily btc/usd data using alphavantage
2. Clean data and convert into pandas dataframe and float
3. Define exploratory variables (13 period MA & 30 period MA) that we are using to predict price of btc/usd
4. Create dataframe of exploratory variables and clean NaN values
5. Initialise the 2 feature variables
6. Set up the dependent variable (In this case, value of btc/usd)
7. Set training data to 80% of dataset
8. Split training data and testing data
9. Train linear regression model
10. Use model to predict test data
11. Use matplotlib to plot predicted price vs actual price
12. Add actual price to predicted price dataframe
13. Calculate r2 score.

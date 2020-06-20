
# Name: Tony Ramsett
# Date started: 5/10/2020
# Project: Telco Customer Churn 
# Class: MSDS692 - Data Practicum 1
# Data Set Reference: (Fendarkar, 2018)

library("C50") # Used for decision trees
library("rpart") # Used for decision trees, pruning, etc.
library("rpart.plot") # Used for plotting within rpart
library("randomForest") # Used for random forest models
library("caret") # Used for correlations and variable importance
library("gmodels") # Used for CrossTable() function
library("plyr") # Used for cleaning data
library("corrplot") # Used for correlation plots
library("adabag") # Used for cross validation on random forest models

# Import data set to R (Data set comes from kaggle.com)
churn <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv", header = T)

# EDA of data set
summary(churn)
str(churn)
head(churn)
dim(churn)

# Review monthly charges in plot, see distribution
summary(churn$MonthlyCharges)
boxplot(churn$MonthlyCharges, main = "Monthly Charges Plot", ylab = "Charges")

# Review summary of churn and tenure
summary(churn$Churn)
summary(churn$tenure)

# Histogram of tenure, feature engineer it later on in code
hist(churn$tenure, main = "Tenure Distribution", xlab = "Tenure", col = c("blue","green")) # blue/green for visibility

# Remove NA's from data set - no need since removing column with all 11 NA's
sum(is.na(churn))
churn[!complete.cases(churn),]
churn <- na.omit(churn)
dim(churn)
sum(is.na(churn))

# Data wrangle - models were not cooperating (logistic regression), map values of 'No Internet Service' to 'No' (now logistic regression runs properly)
# References: (Parimi, S. 2018)
churn$MultipleLines <- as.factor(mapvalues(churn$MultipleLines, from = c("No phone service"), to = c("No")))
churn$OnlineSecurity <- as.factor(mapvalues(churn$OnlineSecurity, from = c("No internet service"), to = c("No")))
churn$OnlineBackup <- as.factor(mapvalues(churn$OnlineBackup, from = c("No internet service"), to = c("No")))
churn$DeviceProtection <- as.factor(mapvalues(churn$DeviceProtection, from = c("No internet service"), to = c("No")))
churn$TechSupport <- as.factor(mapvalues(churn$TechSupport, from = c("No internet service"), to = c("No")))
churn$StreamingTV <- as.factor(mapvalues(churn$StreamingTV, from = c("No internet service"), to = c("No")))
churn$StreamingMovies <- as.factor(mapvalues(churn$StreamingMovies, from = c("No internet service"), to = c("No")))

# Focus solely on plan details, not demographics of customers - remove columns
churn$customerID <- NULL
churn$gender <- NULL
churn$Partner <- NULL
churn$Dependents <- NULL
churn$SeniorCitizen <- NULL
churn$TotalCharges <- NULL

# Feature engineer tenure into groups
# References: (Parimi, S. 2018), (TutorialsPoint, 2020)
tenure_function <- function(tenure_new){
   if(tenure_new >= 0 & tenure_new <= 12) 
    { return('First Year') }
   else if(tenure_new > 12 & tenure_new <= 24) 
    { return('Second Year') }
   else if(tenure_new > 24 & tenure_new <= 36) 
    { return('Third Year') }
   else if(tenure_new > 36 & tenure_new <= 48) 
    { return('Fourth Year') }
   else if(tenure_new > 48 & tenure_new <= 60) 
    { return('Fifth Year') }
   else if(tenure_new > 60 & tenure_new <= 72) 
    { return('Sixth Year') }
}

# Prepare new tenure groups for models, remove old tenure columns
# References: (Parimi, S. 2018)
churn$tenure_year <- sapply(churn$tenure, tenure_function)
churn$tenure_year <- as.factor(churn$tenure_year)
table(churn$tenure_year)

churn$tenure_year_2 <- as.numeric(churn$tenure_year)
hist(churn$tenure_year_2, main = "Tenure Distribution", xlab = "Tenure", col = "green") # View new distribution of tenure groups
churn$tenure <- NULL
churn$tenure_year_2 <- NULL

# Check data set after cleaning, wrangling, and feature engineering
str(churn)
dim(churn)

# Correlation Plot
# References: (An Introduction to corrplot Package, n.d.), (Braglia, L., 2014), & (Ramsett, 2019))

# Prepare correlation plot
churn_corr <- churn
churn_cols <- c(1:15)

# Convert to numeric data types to run plot
churn_corr[churn_cols] <- sapply(churn_corr[churn_cols],as.numeric)
sapply(chrun_corr, class)

# Create correlation plot
churn_corrplot <- cor(churn_corr)
corrplot(churn_corrplot, method = "circle", type = "upper")

# Create training and test data sets, 75/25 split
set.seed(135)
sample_churn <- sample(nrow(churn), 0.75 * nrow(churn)) #75/25 split
churn_train <- churn[sample_churn, ]
churn_test <- churn[-sample_churn, ]

# Confirm splits worked correctly with 75% splits of Churn classifier
# Refernces: (Ramsett, 2019)

dim(churn_train); dim(churn_test)
prop.table(table(churn_train$Churn)); prop.table(table(churn_test$Churn))

# 1 - Logistic Regression
# References: (Saraswat, M., n.d.)
set.seed(135) # Seed set for reproduceble results

# Create model and view results
churn_lr <- glm(Churn ~ ., family = binomial, data = churn_train)
summary(churn_lr)

# Variable Importance output
varImp(churn_lr)

# Create prediction model with success rate output
predict_churn <- predict(churn_lr, churn_test, type = "response")
CrossTable(churn_test$Churn, predict_churn > 0.5, dnn = c("Actual Churn", "Predicted Churn"))

# 10 Fold Cross Validation
# References: (Saraswat, M., n.d.)
set.seed(135) # Seed set for reproduceble results

# Train control - cross-validation with 10 folds
train_control <- trainControl(method = "cv", number = 10)

# Fit logistic regression Model
set.seed(135) # Seed set for reproduceble results

# Create cross-validation model
model <- train(Churn ~ ., data = churn_test, trControl = train_control, method = "glm", family = binomial)

# View model results and accuracy
print(model)
confusionMatrix(model, "none")

# 2 - Decision Trees
# References: (Lantz, 2019), (Ramsett, 2019)
set.seed(135) # Seed set for reproduceble results

# Create decision tree model, revieww results
churn_dt_model <- rpart(Churn ~ ., data = churn_train, method = "class")
summary(churn_dt_model)
plotcp(churn_dt_model)
print(churn_dt_model)

# Plot the tree
rpart.plot(churn_dt_model, digits = 2, fallen.leaves = TRUE, type = 3, extra = 106) # p.209 - Lantz

# Create decision tree model and view summary of prediction model
churn_dt_predict<- predict(churn_dt_model, churn_test, type = "class")
summary(churn_dt_predict)

# Accuracy of model
CrossTable(churn_dt_predict, churn_test[,14], dnn = c("Actual Churn", "Predicted Churn"))
mean(churn_dt_predict == churn_test[,14])

# Pruned Decision Trees
# References: (Das, S., 2017), (Lantz, 2019)
set.seed(135) # Seed set for reproduceble results

# Create model and view results
churn_dt_pruned <- prune(churn_dt_model, cp = 0.03)
summary(churn_dt_pruned)

rpart.plot(churn_dt_pruned, digits = 2, fallen.leaves = TRUE, type = 3, extra = 106) # p.209 - Lantz

# Create prediction model and view results/accuracy
test_pred <- predict(churn_dt_pruned, churn_test, type = "class")
summary(test_pred)

post_prune_acc <- mean(test_pred == churn_test[,14])
print(post_prune_acc)

CrossTable(test_pred, churn_test[,14], dnn = c("Actual Churn", "Predicted Churn"))

# 3 - Random Forest
# References: (Lantz, 2019)
set.seed(135) # Seed set for reproduceble results

# Create model and view results
churn_rf <- randomForest(Churn ~ ., data = churn_train, importance = T)
summary(churn_rf)
print(churn_rf)

# Create prediction model and view results/accuracy
churn_rf_predict <- predict(churn_rf, newdata = churn_test)
mean(churn_rf_predict == churn_test[,14])

CrossTable(churn_rf_predict, churn_test[,14], dnn = c("Actual Churn", "Predicted Churn"))

# Variable Importance Plot
varImpPlot(churn_rf)

# Random Forest model with 10 fold Cross Validation
set.seed(135) # Seed set for reproduceble results

# Create model
churn_boost <- boosting.cv(Churn ~ ., data = churn_test)
summary(churn_boost)

# View results/accuracy of model
churn_boost$confusion

# 4 - Calculate Savings Model
# References: (DataOptimal, 2020), (Heintz, 2018)

# Cross table from logistic regression model (most accurate ML model, use it for cost savings model)
CrossTable(churn_test$Churn, predict_churn > 0.5, dnn = c("Actual Churn", "Predicted Churn"))

# Outputs from cross table of logistic regression model
TN <- 1187
FP <- 222
FN <- 115
TP <- 234

# Number of customers in test data set
test_population <- 1758

# Costs per customer
TN_cost <- TN/test_population
FP_cost <- FP/test_population
FN_cost <- FN/test_population
TP_cost <- TP/test_population

# Model savings per customer total
model_customer_cost <- round((TN_cost * 0) + (FP_cost * 50) + (FN_cost * 250) + (TP_cost * 50), 2)
model_customer_cost

# Apply model cost to entire data set
model_idea <- model_customer_cost * 7043
model_idea

# Business Idea #1 - Spend $50 on everyone for Retention program
bus_idea_1 <- 50 * 7043
bus_idea_1

# Business Idea #2 - Do nothing (Spend nothing on retained customers, lose $250 per churned customer)
bus_idea_2 <- ((1301 * 0) + (1869 * 250))
bus_idea_2

# Plot for quick visual comparison of model versus business idea #1 and #2
barplot_costs <- c(model_idea, bus_idea_1, bus_idea_2)
barplot(barplot_costs, names.arg = barplot_costs, col = c("blue", "red", "green"),
        legend = c("Model Costs", "Retain Everyone", "Remain as is"))

excel_path <- file.choose()
library(xlsx)
library (ggplot2)
library(caret)
library(rpart)

#upload variables from excel sheet
Sleep_Quality  <- as.numeric(unlist(read.xlsx(excel_path, sheetName = "sleepdata_2", colIndex = 5)))

Regularity <- as.numeric(unlist(read.xlsx(excel_path, sheetName = "sleepdata_2", colIndex = 6)))
Steps_Next_Day <- as.numeric(unlist(read.xlsx(excel_path, sheetName = "sleepdata_2", colIndex = 9)))
Movements_per_hour <- as.numeric(unlist(read.xlsx(excel_path, sheetName = "sleepdata_2", colIndex = 13)))
Time_in_bed <- as.numeric(unlist(read.xlsx(excel_path, sheetName = "sleepdata_2", colIndex = 14)))
Time_asleep <- as.numeric(unlist(read.xlsx(excel_path, sheetName = "sleepdata_2", colIndex = 15)))
Time_before_sleep <- as.numeric(unlist(read.xlsx(excel_path, sheetName = "sleepdata_2", colIndex = 16)))

#Create data frame
DataFrame <- data.frame(Sleep_Quality, Regularity, Steps_Next_Day, Movements_per_hour, Time_in_bed, Time_asleep, Time_before_sleep)

# Create extra feature 'Sleep Efficiency' in order to support data and add it to the data frame
DataFrame$Sleep_Efficiency <- DataFrame$Time_asleep / DataFrame$Time_in_bed

# Remove missing values
DataFrame <- na.omit(DataFrame)

# Calculate the correlation matrix
cor_matrix <- cor(DataFrame)

# Plot the heat map
library(ggplot2)
library(ggcorrplot)

ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower",
           lab = TRUE, lab_size = 3)

# Create data frame 2 removing Steps_next_day and Movements_per_hour to optimize machine learning model
DataFrame2 <- data.frame(Sleep_Quality, Time_in_bed, Time_asleep, DataFrame$Sleep_Efficiency, Regularity)

# Standardized the values of DataFrame2 between 0 and 1
DataFrame_standardized_2 <- as.data.frame(scale(DataFrame2))

# Load the necessary libraries for machine learning algorithm
library(xgboost)

# Set the seed for reproducibility
set.seed(123)

# Split the data into training and testing sets (80/20 split)
trainIndex <- createDataPartition(DataFrame_standardized_2$Sleep_Quality, p = .8, list = FALSE, times = 1)
trainData <- DataFrame_standardized_2[trainIndex, ]
testData <- DataFrame_standardized_2[-trainIndex, ]

# Define the training control parameters for the models
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = "final")

# Train several machine learning models on the data
model1 <- train(Sleep_Quality ~ ., data = trainData, method = "lm", trControl = ctrl)
model2 <- train(Sleep_Quality ~ ., data = trainData, method = "rf", trControl = ctrl)
model3 <- train(Sleep_Quality ~ ., data = trainData, method = "gbm", trControl = ctrl)

# Evaluate the performance of the models on the testing data
predictions1 <- predict(model1, newdata = testData)
RMSE1 <- sqrt(mean((testData$Sleep_Quality - predictions1)^2))
predictions2 <- predict(model2, newdata = testData)
RMSE2 <- sqrt(mean((testData$Sleep_Quality - predictions2)^2))
predictions3 <- predict(model3, newdata = testData)
RMSE3 <- sqrt(mean((testData$Sleep_Quality - predictions3)^2))

# Display the performance of all models
data.frame(Model = c("Linear Regression", "Random Forest", "Gradient Boosting"), 
           RMSE = c(RMSE1, RMSE2, RMSE3))
# Create a data frame with the RMSE values for each model
rmse_df <- data.frame(Model = c("Linear Regression", "Random Forest", "Gradient Boosting"),
                      RMSE = c(RMSE1, RMSE2, RMSE3))

# Create a bar plot of the RMSE values
ggplot(rmse_df, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  scale_fill_discrete(name = "Model") +
  labs(x = "Model", y = "RMSE", title = "RMSE for Sleep Quality Prediction Models") +
  theme(plot.title = element_text(hjust = 0.5))




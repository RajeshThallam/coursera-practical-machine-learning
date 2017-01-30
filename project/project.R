train.raw <- read.csv('./data/pml-training.csv')
test.raw <- read.csv('./data/pml-testing.csv')

# Data Exploration
head(train.raw)
summary(train.raw)[10]

dim(train.raw)
names(train.raw)

colSums(is.na(train.raw) | train.raw == '')

#### Check for features's variance

# Based on the principal component analysis PCA, it is important that features have maximum variance for maximum uniqueness, so that each feature is as distant as possible (as orthogonal as possible) from the other features.   

# check for zero variance
zero.var = nearZeroVar(train, saveMetrics=TRUE)
zero.var

# There is no features without variability (all has enough variance). So there is no feature to be removed further.  

# Data Cleansing
maxNAPerc = 20
maxNACount <- nrow(train.raw) / 100 * maxNAPerc


removeColumns <- which(colSums(is.na(train.raw) | train.raw=="") > maxNACount)
train.01 <- train.raw[,-removeColumns]
test.01 <- test.raw[,-removeColumns]


# Also remove all time related data, since we won't use those


removeColumns <- grep("timestamp|window", names(train.01))
removeColumns

train.02 <- train.01[,-c(1, removeColumns )]
test.02 <- test.01[,-c(1, removeColumns )]

# Then convert all factors to integers

classeLevels <- levels(train.02$classe)
train.03 <- data.frame(data.matrix(train.02))
train.03$classe <- factor(train.03$classe, labels=classeLevels)
test.03 <- data.frame(data.matrix(test.02))

# Feature Selection
train.cln <- train.03
test.cln <- test.03



set.seed(118258)
library(caret)

classeIndex <- which(names(train.cln) == "classe")

partition <- createDataPartition(y=train.cln$classe, p=0.75, list=FALSE)
train <- train.cln[partition, ]
validation <- train.cln[-partition, ]

#Plot a correlation matrix between features.   
#A good set of features is when they are highly uncorrelated (orthogonal) each others. The plot below shows average of correlation is not too high, so I choose to not perform further PCA preprocessing.   

#```{r fig.width=12, fig.height=12, dpi=72}
correlations <- cor(train[, -classeIndex], as.numeric(train$classe))
bestCorrelations <- subset(as.data.frame(as.table(correlations)), abs(Freq)>0.2)
bestCorrelations

library(Rmisc)
library(ggplot2)

p1 <- ggplot(train, aes(classe,pitch_forearm)) + geom_boxplot(aes(fill=classe))
p2 <- ggplot(train, aes(classe, magnet_arm_x)) + geom_boxplot(aes(fill=classe))
p3 <- ggplot(train, aes(classe, magnet_arm_y)) + geom_boxplot(aes(fill=classe))
p4 <- ggplot(train, aes(classe, accel_arm_x)) + geom_boxplot(aes(fill=classe))
p5 <- ggplot(train, aes(classe, magnet_belt_y)) + geom_boxplot(aes(fill=classe))

multiplot(p1,p2,p3,p4,p5,cols=3, rows=2)

library(corrplot)
correlationMatrix <- cor(train[, -classeIndex])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9, exact=TRUE)
excludeColumns <- c(highlyCorrelated, classeIndex)
corrplot(correlationMatrix, method="color", type="lower", order="hclust", tl.cex=0.70, tl.col="black", tl.srt = 45, diag = FALSE)

pcaPreProcess.all <- preProcess(train[, -classeIndex], method = "pca", thresh = 0.99)
train.pca.all <- predict(pcaPreProcess.all, train[, -classeIndex])
validation.pca.all <- predict(pcaPreProcess.all, validation[, -classeIndex])
test.pca.all <- predict(pcaPreProcess.all, test.cln[, -classeIndex])


pcaPreProcess.subset <- preProcess(train[, -excludeColumns], method = "pca", thresh = 0.99)
train.pca.subset <- predict(pcaPreProcess.subset, train[, -excludeColumns])
validation.pca.subset <- predict(pcaPreProcess.subset, validation[, -excludeColumns])
test.pca.subset <- predict(pcaPreProcess.subset, test.cln[, -classeIndex])

# Model Builiding
control <- trainControl(method = "cv", number = 5)
fit_rpart <- train(classe ~ ., data = train, method = "rpart", trControl = control)
print(fit_rpart, digits = 4)

library(rattle)
library(rpart)
library(repmis)
fancyRpartPlot(fit_rpart$finalModel)

modFitA1 <- rpart(train$classe ~ ., data = train[, -excludeColumns], method="class")
fancyRpartPlot(modFitA1)

predictionsA1 <- predict(modFitA1, validation[, -excludeColumns], type = "class")
confusionMatrix(predictionsA1, validation$classe)

# Model Selection

# Model Fit for Testing

# Conclusion

help(names)
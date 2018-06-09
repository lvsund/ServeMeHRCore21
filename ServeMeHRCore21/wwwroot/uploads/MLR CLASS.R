# Get the data
MYdataset <- read.csv("Y:/Documents/bookcode/bookcode/jobclassinfo2.csv")
#MYdataset <- read.csv("/Volumes/NAS/R Files/People HR Analytics Book - Copy/jobclassinfo2.txt")
#look at structure
str(MYdataset, width = 80, strict.width = "wrap")

#Ensure all needed libraries are installed
list.of.packages <- c("plyr", "dplyr",  "ROCR", "caret", "randomForest",
                      "kernlab", "magrittr", "rpart", "ggplot2", "nnet", "car",
                      "rpart.plot", "pROC", "ada", "readr","mlr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[, "Package"])]
if (length(new.packages))
  install.packages(new.packages)


#look at data summary
summary(MYdataset)
#library(magrittr) # For the %>% and %<>% operators.



# A pre-defined value is used to reset the random seed so that results are repeatable.

seed <- 42


set.seed(seed)

#Most of the time we want to carve data into testing and training datasets. the following code does that
#but since for this data we need all cases we will still use whole dataset MYdataset.

MYnobs <- nrow(MYdataset) # 66 observations 
MYsample <- MYtrain <- sample(nrow(MYdataset), 0.7 * MYnobs) # 46 observations
MYvalidate <- sample(setdiff(seq_len(nrow(MYdataset)), MYtrain), 0.14 * MYnobs) # 9 observations
MYtest <- setdiff(setdiff(seq_len(nrow(MYdataset)), MYtrain), MYvalidate) # 11 observations



#============================================================
# lets organize our data into various labels for easy coding

MYinput <- c("EducationLevel", "Experience", "OrgImpact", "ProblemSolving",
             "Supervision", "ContactLevel", "FinancialBudget")

MYnumeric <- c("EducationLevel", "Experience", "OrgImpact", "ProblemSolving",
               "Supervision", "ContactLevel", "FinancialBudget")

MYcategoric <- NULL

MYtarget <- "PG"
MYrisk <- NULL
MYident <- "ID"
MYignore <- c("JobFamily", "JobFamilyDescription", "JobClass", "JobClassDescription", "PayGrade")
MYweights <- NULL

MYTrainingData <- MYdataset[c(MYinput, MYtarget)]
MYTestingData <- MYdataset[c(MYinput, MYtarget)]

library(mlr)

## Specify the type of analysis (e.g. classification) and provide data and response variable
task = makeClassifTask(data = MYTrainingData, target = "PG")

## 2) Define the learner
## Choose a specific algorithm (e.g. linear discriminant analysis)
lrn = makeLearner("classif.randomForest", predict.type = "response", fix.factors.prediction = TRUE)
model = train(lrn, task)

pred = predict(model, newdata = MYTestingData )
pred
data.frame(pred)
dfpred<-data.frame(cbind(MYTestingData,pred))
dfpred
performance(pred, measures = list(mmce, acc))

calculateConfusionMatrix(pred, relative = TRUE, sums = FALSE)

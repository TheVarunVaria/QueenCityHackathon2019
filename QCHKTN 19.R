setwd("/Users/thevarunvaria/Downloads/qc-hackathon/substance-abuse/data/")
mydata=read.csv("HackTrain.csv", sep=",", 
                header=T, strip.white=T,
                na.strings=c("NA", "NaN", "", "?"))

mydata$CASEID=NULL
mydata$YEAR=NULL
mydata$DISYR=NULL
mydata$DETNLF=NULL
mydata$PREG=NULL
mydata$DETCRIM=NULL
mydata$ROUTE3=NULL
mydata$FREQ3=NULL
mydata$FRSTUSE3=NULL
mydata$IDU=NULL
mydata$DSMCRIT=NULL
mydata$CBSA=NULL
mydata$DAYWAIT=NULL
mydata$ROUTE2=NULL
mydata$FREQ2=NULL
mydata$FRSTUSE2=NULL
mydata$ROUTE3=NULL
mydata$FREQ3=NULL
mydata$FRSTUSE3=NULL
mydata$IDU=NULL
mydata$HLTHINS=NULL
mydata$PRIMPAY=NULL
mydata$EDUC=NULL
mydata$PSOURCE=NULL
mydata$SUB1=NULL



mydata[mydata==-9]<-NA

colSums(is.na(mydata))## counting missing columns

#dropping some rows
mydata <- mydata[complete.cases(mydata$SERVSETA),]
mydata <- mydata[complete.cases(mydata$GENDER),]
mydata <- mydata[complete.cases(mydata$LOS),]
mydata <- mydata[complete.cases(mydata$SERVSETD),]
mydata <- mydata[complete.cases(mydata$SUB1),]

mydata$RACE[is.na(mydata$RACE)]<- 5
mydata$ETHNIC[is.na(mydata$ETHNIC )]<- 5
mydata$MARSTAT [is.na(mydata$MARSTAT)]<- 1
mydata$EDUC [is.na(mydata$EDUC)]<- 3
mydata$EMPLOY [is.na(mydata$EMPLOY)]<-runif(3,4)
mydata$VET [is.na(mydata$VET)]<- 2
mydata$LIVARAG [is.na(mydata$LIVARAG)]<- 3
mydata$PRIMINC [is.na(mydata$PRIMINC)]<- runif(1,21) ## check again
mydata$ARRESTS [is.na(mydata$ARRESTS)]<-0
mydata$METHUSE [is.na(mydata$METHUSE)]<-2
mydata$PSOURCE [is.na(mydata$PSOURCE)]<-runif(1,7)
mydata$NOPRIOR [is.na(mydata$NOPRIOR)]<-0
mydata$ROUTE1 [is.na(mydata$ROUTE1)]<-1
mydata$FREQ1 [is.na(mydata$FREQ1)]<-5
mydata$FRSTUSE1 [is.na(mydata$FRSTUSE1)]<-3
mydata$SUB2 [is.na(mydata$SUB2)]<-1
mydata$SUB3 [is.na(mydata$SUB3)]<-1
mydata$PSYPROB [is.na(mydata$PSYPROB)]<-2

#mydata<- subset(mydata, select=-c(METHFLG,DISYR, AMPHFLG, STIMFLG, TRNQFLG, STIMFLG, TRNQFLG, BARBFLG,SEDHPFLG,INHFLG,OTCFLG))
# mydata$AGE <- as.factor(mydata$AGE)
# mydata$GENDER <- as.factor(mydata$GENDER)
# mydata$RACE <- as.factor(mydata$RACE)
# mydata$ETHNIC <- as.factor(mydata$ETHNIC)
# mydata$MARSTAT <- as.factor(mydata$MARSTAT)
# mydata$EDUC<-as.factor(mydata$EDUC)
# mydata$EMPLOY  <-as.factor(mydata$EMPLOY)
# mydata$DETNLF <- as.factor(mydata$DETNLF)
# mydata$PREG <- as.factor(mydata$PREG)
# mydata$VET <- as.factor(mydata$VET)
# mydata$LIVARAG <- as.factor(mydata$LIVARAG)
# mydata$PRIMINC <- as.factor(mydata$PRIMINC)
# mydata$ARRESTS <- as.factor(mydata$ARRESTS)
# mydata$STFIPS <- NULL
# mydata$CBSA <- NULL
# mydata$REGION <- as.factor(mydata$REGION)
# mydata$DIVISION <- as.factor(mydata$DIVISION)
# mydata$SERVSETD <- as.factor(mydata$SERVSETD)
# mydata$METHUSE <- as.factor(mydata$METHUSE)
# mydata$REASON <- as.factor(mydata$REASON)
# mydata$PSOURCE <- as.factor(mydata$PSOURCE)
# mydata$DETCRIM <- as.factor(mydata$DETCRIM)
# mydata$NOPRIOR <- as.factor(mydata$NOPRIOR)
# mydata$SUB1 <- as.factor(mydata$SUB1)
# mydata$ROUTE1 <- as.factor(mydata$ROUTE1)
# mydata$FREQ1 <- as.factor(mydata$FREQ1)
# mydata$FRSTUSE1 <- as.factor(mydata$FRSTUSE1)
# mydata$SUB2 <- as.factor(mydata$SUB2)
# mydata$ROUTE2 <- as.factor(mydata$ROUTE2)
# mydata$FREQ2 <- as.factor(mydata$FREQ2)
# mydata$FRSTUSE2 <- as.factor(mydata$FRSTUSE2)
# mydata$SUB3 <- as.factor(mydata$SUB3)
# mydata$ROUTE3 <- as.factor(mydata$ROUTE3)
# mydata$FREQ3 <- as.factor(mydata$FREQ3)
# mydata$FRSTUSE3 <- as.factor(mydata$FRSTUSE3)
# mydata$PREG <- NULL

mydata$SERVSETA <- as.factor(mydata$SERVSETA)
mydata$AGE <- as.factor(mydata$AGE)
mydata$GENDER <- as.factor(mydata$GENDER)
mydata$RACE <- as.factor(mydata$RACE)
mydata$ETHNIC <- as.factor(mydata$ETHNIC)
mydata$MARSTAT <- as.factor(mydata$MARSTAT)
mydata$EDUC<-as.factor(mydata$EDUC)
mydata$EMPLOY  <-as.factor(mydata$EMPLOY)
#mydata$DETNLF <- as.factor(mydata$DETNLF)
#mydata$PREG <- as.factor(mydata$PREG)
mydata$VET <- as.factor(mydata$VET)
mydata$LIVARAG <- as.factor(mydata$LIVARAG)
mydata$PRIMINC <- as.factor(mydata$PRIMINC)
mydata$ARRESTS <- as.factor(mydata$ARRESTS)
mydata$STFIPS <- as.factor(mydata$STFIPS)
#mydata$CBSA <- as.factor(mydata$CBSA)
mydata$REGION <- as.factor(mydata$REGION)
mydata$DIVISION <- as.factor(mydata$DIVISION)
mydata$SERVSETD <- as.factor(mydata$SERVSETD)
mydata$METHUSE <- as.factor(mydata$METHUSE)
mydata$REASON <- as.factor(mydata$REASON)
mydata$PSOURCE <- as.factor(mydata$PSOURCE)
#mydata$DETCRIM <- as.factor(mydata$DETCRIM)
mydata$NOPRIOR <- as.factor(mydata$NOPRIOR)
mydata$SUB1 <- as.factor(mydata$SUB1)
mydata$ROUTE1 <- as.factor(mydata$ROUTE1)
mydata$FREQ1 <- as.factor(mydata$FREQ1)
mydata$FRSTUSE1 <- as.factor(mydata$FRSTUSE1)
mydata$SUB2 <- as.factor(mydata$SUB2)
#mydata$ROUTE2 <- as.factor(mydata$ROUTE2)
#mydata$FREQ2 <- as.factor(mydata$FREQ2)
#mydata$FRSTUSE2 <- as.factor(mydata$FRSTUSE2)
mydata$SUB3 <- as.factor(mydata$SUB3)
#mydata$ROUTE3 <- as.factor(mydata$ROUTE3)
#mydata$FREQ3 <- as.factor(mydata$FREQ3)
#mydata$FRSTUSE3 <- as.factor(mydata$FRSTUSE3)
mydata$NUMSUBS <- as.factor(mydata$NUMSUBS)
mydata$IDU <- as.factor(mydata$IDU)
mydata$ALCFLG <- as.factor(mydata$ALCFLG)
mydata$COKEFLG <- as.factor(mydata$COKEFLG)
mydata$MARFLG <- as.factor(mydata$MARFLG)
mydata$HERFLG <- as.factor(mydata$HERFLG)
mydata$METHFLG <- as.factor(mydata$METHFLG)
mydata$OPSYNFLG <- as.factor(mydata$OPSYNFLG)
mydata$PCPFLG <- as.factor(mydata$PCPFLG)
mydata$HALLFLG <- as.factor(mydata$HALLFLG)
mydata$MTHAMFLG <- as.factor(mydata$MTHAMFLG)
mydata$AMPHFLG <- as.factor(mydata$AMPHFLG)
mydata$STIMFLG <- as.factor(mydata$STIMFLG)
mydata$BENZFLG <- as.factor(mydata$BENZFLG)
mydata$TRNQFLG <- as.factor(mydata$TRNQFLG)
mydata$BARBFLG <- as.factor(mydata$BARBFLG)
mydata$SEDHPFLG <- as.factor(mydata$SEDHPFLG)
mydata$INHFLG <- as.factor(mydata$INHFLG)
mydata$OTCFLG <- as.factor(mydata$OTCFLG)
mydata$OTHERFLG <- as.factor(mydata$OTHERFLG)
mydata$ALCDRUG <- as.factor(mydata$ALCDRUG)
#mydata$DSMCRIT <- as.factor(mydata$DSMCRIT)
mydata$PSYPROB <- as.factor(mydata$PSYPROB)
#mydata$HLTHINS <- as.factor(mydata$HLTHINS)
#mydata$PRIMPAY <- as.factor(mydata$PRIMPAY)

summary(mydata)

#set.seed(1234)
indexes = sample(1:nrow(mydata), size=0.02*nrow(mydata))
nrow(mydata) # Total number of records
mydata.train=mydata[indexes,]
nrow(mydata.train) # Number of records in train
mydata.test=mydata[-indexes,]
nrow(mydata.test) # Number of records in test

library(randomForest)


library("nnet")
summary(mydata.train)


rf <-randomForest(LOS~., data=mydata.train, ntree=40, na.action=na.exclude, importance=T,
                  proximity=F,mtry = 6) 
print(rf)


varImpPlot(rf)
library(mlbench)
library(caret)
importance <- varImp(rf, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)
library(nnet)
set.seed(32) 

train_data <- train_data[complete.cases(train_data),]


#set.seed(1234)
indexes = sample(1:nrow(mydata), size=0.4*nrow(mydata))
nrow(mydata) # Total number of records
mydata.train=mydata[indexes,]
nrow(mydata.train) # Number of records in train
mydata.train <- mydata.train[complete.cases(mydata.train),]

ann <- nnet(LOS ~ ., data=mydata.train, size=10, maxit=500,MaxNWts = 2322)
print(ann)
test_data = tail(mydata, n = 90000) 
test_data <- test_data[complete.cases(test_data),] # Remove missing values from test_da
predicted_values <- predict(ann, test_data,type= "raw") # Use the gbm classifier to make the predictions
final_data <- cbind(test_data, predicted_values) # Add the predictions to test_data
colnames <- c(colnames(test_data),"prob.one") # Add the new column names to the original column names 
library(ROCR)
library(ggplot2)
library(class)
library(caret)
#predicted_values <- predict(knnFit, test_data, type= "prob") # Use the classifier to make the predictions


## Model Evaluation:
head(predicted_values)
threshold <- 0.5
pred <- factor( ifelse(predicted_values[,1] > threshold, 1, 0))
head(pred)
levels(pred)
levels(test_data$salary.class)
confusionMatrix(pred, test_data$REASON, 
                positive = levels(test_data$salary.class)[2])

#predicted_values <- predict(knnFit, test_data, type= "prob") # Use the classifier to make the predictions
pred <- prediction(predicted_values[,1], test_data$REASON)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="Decision Tree")

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))


setwd("/Users/thevarunvaria/Downloads/qc-hackathon/substance-abuse/data/")
mydata2=read.csv("HackTest.csv", sep=",", 
                header=T, strip.white=T,
                na.strings=c("NA", "NaN", "", "?"))
mydata2$CASEID=NULL
mydata2$YEAR=NULL
mydata2$DISYR=NULL
mydata2$DETNLF=NULL
mydata2$PREG=NULL
mydata2$DETCRIM=NULL
mydata2$ROUTE3=NULL
mydata2$FREQ3=NULL
mydata2$FRSTUSE3=NULL
mydata2$IDU=NULL
mydata2$DSMCRIT=NULL
mydata2$CBSA=NULL
mydata2$DAYWAIT=NULL
mydata2$ROUTE2=NULL
mydata2$FREQ2=NULL
mydata2$FRSTUSE2=NULL
mydata2$ROUTE3=NULL
mydata2$FREQ3=NULL
mydata2$FRSTUSE3=NULL
mydata2$IDU=NULL
mydata2$HLTHINS=NULL
mydata2$PRIMPAY=NULL
mydata2$EDUC=NULL
mydata2$PSOURCE=NULL
mydata2$SUB1=NULL

mydata2[mydata2==-9]<-NA

colSums(is.na(mydata2))## counting missing columns

#dropping some rows
# mydata2 <- mydata2[complete.cases(mydata2$SERVSETA),]
# mydata2 <- mydata2[complete.cases(mydata2$GENDER),]
# mydata2 <- mydata2[complete.cases(mydata2$LOS),]
# mydata2 <- mydata2[complete.cases(mydata2$SERVSETD),]
# mydata2 <- mydata2[complete.cases(mydata2$SUB1),]

mydata2$RACE[is.na(mydata2$RACE)]<- 5
mydata2$ETHNIC[is.na(mydata2$ETHNIC )]<- 5
mydata2$MARSTAT [is.na(mydata2$MARSTAT)]<- 1
mydata2$EDUC [is.na(mydata2$EDUC)]<- 3
mydata2$EMPLOY [is.na(mydata2$EMPLOY)]<-runif(3,4)
mydata2$VET [is.na(mydata2$VET)]<- 2
mydata2$LIVARAG [is.na(mydata2$LIVARAG)]<- 3
mydata2$PRIMINC [is.na(mydata2$PRIMINC)]<- runif(1,21) ## check again
mydata2$ARRESTS [is.na(mydata2$ARRESTS)]<-0
mydata2$METHUSE [is.na(mydata2$METHUSE)]<-2
mydata2$PSOURCE [is.na(mydata2$PSOURCE)]<-runif(1,7)
mydata2$NOPRIOR [is.na(mydata2$NOPRIOR)]<-0
mydata2$ROUTE1 [is.na(mydata2$ROUTE1)]<-1
mydata2$FREQ1 [is.na(mydata2$FREQ1)]<-5
mydata2$FRSTUSE1 [is.na(mydata2$FRSTUSE1)]<-3
mydata2$SUB2 [is.na(mydata2$SUB2)]<-1
mydata2$SUB3 [is.na(mydata2$SUB3)]<-1
mydata2$PSYPROB [is.na(mydata2$PSYPROB)]<-2


mydata2$SERVSETA <- as.factor(mydata2$SERVSETA)
mydata2$AGE <- as.factor(mydata2$AGE)
mydata2$GENDER <- as.factor(mydata2$GENDER)
mydata2$RACE <- as.factor(mydata2$RACE)
mydata2$ETHNIC <- as.factor(mydata2$ETHNIC)
mydata2$MARSTAT <- as.factor(mydata2$MARSTAT)
mydata2$EDUC<-as.factor(mydata2$EDUC)
mydata2$EMPLOY  <-as.factor(mydata2$EMPLOY)
#mydata2$DETNLF <- as.factor(mydata2$DETNLF)
#mydata2$PREG <- as.factor(mydata2$PREG)
mydata2$VET <- as.factor(mydata2$VET)
mydata2$LIVARAG <- as.factor(mydata2$LIVARAG)
mydata2$PRIMINC <- as.factor(mydata2$PRIMINC)
mydata2$ARRESTS <- as.factor(mydata2$ARRESTS)
mydata2$STFIPS <- as.factor(mydata2$STFIPS)
#mydata2$CBSA <- as.factor(mydata2$CBSA)
mydata2$REGION <- as.factor(mydata2$REGION)
mydata2$DIVISION <- as.factor(mydata2$DIVISION)
mydata2$SERVSETD <- as.factor(mydata2$SERVSETD)
mydata2$METHUSE <- as.factor(mydata2$METHUSE)
mydata2$REASON <- as.factor(mydata2$REASON)
mydata2$PSOURCE <- as.factor(mydata2$PSOURCE)
#mydata2$DETCRIM <- as.factor(mydata2$DETCRIM)
mydata2$NOPRIOR <- as.factor(mydata2$NOPRIOR)
mydata2$SUB1 <- as.factor(mydata2$SUB1)
mydata2$ROUTE1 <- as.factor(mydata2$ROUTE1)
mydata2$FREQ1 <- as.factor(mydata2$FREQ1)
mydata2$FRSTUSE1 <- as.factor(mydata2$FRSTUSE1)
mydata2$SUB2 <- as.factor(mydata2$SUB2)
#mydata2$ROUTE2 <- as.factor(mydata2$ROUTE2)
#mydata2$FREQ2 <- as.factor(mydata2$FREQ2)
#mydata2$FRSTUSE2 <- as.factor(mydata2$FRSTUSE2)
mydata2$SUB3 <- as.factor(mydata2$SUB3)
#mydata2$ROUTE3 <- as.factor(mydata2$ROUTE3)
#mydata2$FREQ3 <- as.factor(mydata2$FREQ3)
#mydata2$FRSTUSE3 <- as.factor(mydata2$FRSTUSE3)
mydata2$NUMSUBS <- as.factor(mydata2$NUMSUBS)
#mydata2$IDU <- as.factor(mydata2$IDU)
mydata2$ALCFLG <- as.factor(mydata2$ALCFLG)
mydata2$COKEFLG <- as.factor(mydata2$COKEFLG)
mydata2$MARFLG <- as.factor(mydata2$MARFLG)
mydata2$HERFLG <- as.factor(mydata2$HERFLG)
mydata2$METHFLG <- as.factor(mydata2$METHFLG)
mydata2$OPSYNFLG <- as.factor(mydata2$OPSYNFLG)
mydata2$PCPFLG <- as.factor(mydata2$PCPFLG)
mydata2$HALLFLG <- as.factor(mydata2$HALLFLG)
mydata2$MTHAMFLG <- as.factor(mydata2$MTHAMFLG)
mydata2$AMPHFLG <- as.factor(mydata2$AMPHFLG)
mydata2$STIMFLG <- as.factor(mydata2$STIMFLG)
mydata2$BENZFLG <- as.factor(mydata2$BENZFLG)
mydata2$TRNQFLG <- as.factor(mydata2$TRNQFLG)
mydata2$BARBFLG <- as.factor(mydata2$BARBFLG)
mydata2$SEDHPFLG <- as.factor(mydata2$SEDHPFLG)
mydata2$INHFLG <- as.factor(mydata2$INHFLG)
mydata2$OTCFLG <- as.factor(mydata2$OTCFLG)
mydata2$OTHERFLG <- as.factor(mydata2$OTHERFLG)
mydata2$ALCDRUG <- as.factor(mydata2$ALCDRUG)
#mydata2$DSMCRIT <- as.factor(mydata2$DSMCRIT)
mydata2$PSYPROB <- as.factor(mydata2$PSYPROB)
#mydata$HLTHINS <- as.factor(mydata$HLTHINS)
#mydata$PRIMPAY <- as.factor(mydata$PRIMPAY)

predicted_values2 <- predict(ann, mydata2,type= "raw") # Use the gbm classifier to make the predictions
head(predicted_values2)
threshold <- 0.5
pred2 <- factor( ifelse(predicted_values2[,1] > threshold, 1, 0))
head(pred2)
levels(pred2)

write.csv(pred2,file="testdata_QCHKTN")

predicted_values <- predict(rf, mydata2) # Use the rf classifier to make the predictions

rf <-randomForest(LOS~., data=mydata.train, ntree=80, na.action=na.exclude, importance=T,
                  proximity=F,mtry = 8) 






setwd("/Users/thevarunvaria/Downloads/qc-hackathon/substance-abuse/data/")
mydata3=read.csv("HackTest.csv", sep=",", 
                 header=T, strip.white=T,
                 na.strings=c("NA", "NaN", "", "?"))
mydata3$CASEID=NULL
mydata3$YEAR=NULL
mydata3$DISYR=NULL
mydata3$DETNLF=NULL
mydata3$PREG=NULL
mydata3$DETCRIM=NULL
mydata3$ROUTE3=NULL
mydata3$FREQ3=NULL
mydata3$FRSTUSE3=NULL
mydata3$IDU=NULL
mydata3$DSMCRIT=NULL
mydata3$CBSA=NULL
mydata3$DAYWAIT=NULL
mydata3$ROUTE2=NULL
mydata3$FREQ2=NULL
mydata3$FRSTUSE2=NULL
mydata3$ROUTE3=NULL
mydata3$FREQ3=NULL
mydata3$FRSTUSE3=NULL
mydata3$IDU=NULL
mydata3$HLTHINS=NULL
mydata3$PRIMPAY=NULL
mydata3$EDUC=NULL
mydata3$PSOURCE=NULL
mydata3$SUB1=NULL



mydata3[mydata3==-9]<-NA

colSums(is.na(mydata3))## counting missing columns

#dropping some rows
mydata3 <- mydata[complete.cases(mydata$SERVSETA),]
mydata3 <- mydata[complete.cases(mydata$GENDER),]
mydata3 <- mydata[complete.cases(mydata$LOS),]
mydata3 <- mydata[complete.cases(mydata$SERVSETD),]
mydata3 <- mydata[complete.cases(mydata$SUB1),]

mydata3$RACE[is.na(mydata3$RACE)]<- 5
mydata3$ETHNIC[is.na(mydata3$ETHNIC )]<- 5
mydata3$MARSTAT [is.na(mydata3$MARSTAT)]<- 1
mydata3$EDUC [is.na(mydata3$EDUC)]<- 3
mydata3$EMPLOY [is.na(mydata3$EMPLOY)]<-runif(3,4)
mydata3$VET [is.na(mydata3$VET)]<- 2
mydata3$LIVARAG [is.na(mydata3$LIVARAG)]<- 3
mydata3$PRIMINC [is.na(mydata3$PRIMINC)]<- runif(1,21) ## check again
mydata3$ARRESTS [is.na(mydata3$ARRESTS)]<-0
mydata3$METHUSE [is.na(mydata3$METHUSE)]<-2
mydata3$PSOURCE [is.na(mydata3$PSOURCE)]<-runif(1,7)
mydata3$NOPRIOR [is.na(mydata3$NOPRIOR)]<-0
mydata3$ROUTE1 [is.na(mydata3$ROUTE1)]<-1
mydata3$FREQ1 [is.na(mydata3$FREQ1)]<-5
mydata3$FRSTUSE1 [is.na(mydata3$FRSTUSE1)]<-3
mydata3$SUB2 [is.na(mydata3$SUB2)]<-1
mydata3$SUB3 [is.na(mydata3$SUB3)]<-1
mydata3$PSYPROB [is.na(mydata3$PSYPROB)]<-2

#mydata3<- subset(mydata3, select=-c(METHFLG,DISYR, AMPHFLG, STIMFLG, TRNQFLG, STIMFLG, TRNQFLG, BARBFLG,SEDHPFLG,INHFLG,OTCFLG))
# mydata3$AGE <- as.factor(mydata3$AGE)
# mydata3$GENDER <- as.factor(mydata3$GENDER)
# mydata3$RACE <- as.factor(mydata3$RACE)
# mydata3$ETHNIC <- as.factor(mydata3$ETHNIC)
# mydata3$MARSTAT <- as.factor(mydata3$MARSTAT)
# mydata3$EDUC<-as.factor(mydata3$EDUC)
# mydata3$EMPLOY  <-as.factor(mydata3$EMPLOY)
# mydata3$DETNLF <- as.factor(mydata3$DETNLF)
# mydata3$PREG <- as.factor(mydata3$PREG)
# mydata3$VET <- as.factor(mydata3$VET)
# mydata3$LIVARAG <- as.factor(mydata3$LIVARAG)
# mydata3$PRIMINC <- as.factor(mydata3$PRIMINC)
# mydata3$ARRESTS <- as.factor(mydata3$ARRESTS)
# mydata3$STFIPS <- NULL
# mydata3$CBSA <- NULL
# mydata3$REGION <- as.factor(mydata3$REGION)
# mydata3$DIVISION <- as.factor(mydata3$DIVISION)
# mydata3$SERVSETD <- as.factor(mydata3$SERVSETD)
# mydata3$METHUSE <- as.factor(mydata3$METHUSE)
# mydata3$REASON <- as.factor(mydata3$REASON)
# mydata3$PSOURCE <- as.factor(mydata3$PSOURCE)
# mydata3$DETCRIM <- as.factor(mydata3$DETCRIM)
# mydata3$NOPRIOR <- as.factor(mydata3$NOPRIOR)
# mydata3$SUB1 <- as.factor(mydata3$SUB1)
# mydata3$ROUTE1 <- as.factor(mydata3$ROUTE1)
# mydata3$FREQ1 <- as.factor(mydata3$FREQ1)
# mydata3$FRSTUSE1 <- as.factor(mydata3$FRSTUSE1)
# mydata3$SUB2 <- as.factor(mydata3$SUB2)
# mydata3$ROUTE2 <- as.factor(mydata3$ROUTE2)
# mydata3$FREQ2 <- as.factor(mydata3$FREQ2)
# mydata3$FRSTUSE2 <- as.factor(mydata3$FRSTUSE2)
# mydata3$SUB3 <- as.factor(mydata3$SUB3)
# mydata3$ROUTE3 <- as.factor(mydata3$ROUTE3)
# mydata3$FREQ3 <- as.factor(mydata3$FREQ3)
# mydata3$FRSTUSE3 <- as.factor(mydata3$FRSTUSE3)
# mydata3$PREG <- NULL

mydata3$SERVSETA <- as.factor(mydata3$SERVSETA)
mydata3$AGE <- as.factor(mydata3$AGE)
mydata3$GENDER <- as.factor(mydata3$GENDER)
mydata3$RACE <- as.factor(mydata3$RACE)
mydata3$ETHNIC <- as.factor(mydata3$ETHNIC)
mydata3$MARSTAT <- as.factor(mydata3$MARSTAT)
mydata3$EDUC<-as.factor(mydata3$EDUC)
mydata3$EMPLOY  <-as.factor(mydata3$EMPLOY)
#mydata3$DETNLF <- as.factor(mydata3$DETNLF)
#mydata3$PREG <- as.factor(mydata3$PREG)
mydata3$VET <- as.factor(mydata3$VET)
mydata3$LIVARAG <- as.factor(mydata3$LIVARAG)
mydata3$PRIMINC <- as.factor(mydata3$PRIMINC)
mydata3$ARRESTS <- as.factor(mydata3$ARRESTS)
mydata3$STFIPS <- as.factor(mydata3$STFIPS)
#mydata3$CBSA <- as.factor(mydata3$CBSA)
mydata3$REGION <- as.factor(mydata3$REGION)
mydata3$DIVISION <- as.factor(mydata3$DIVISION)
mydata3$SERVSETD <- as.factor(mydata3$SERVSETD)
mydata3$METHUSE <- as.factor(mydata3$METHUSE)
mydata3$REASON <- as.factor(mydata3$REASON)
mydata3$PSOURCE <- as.factor(mydata3$PSOURCE)
#mydata3$DETCRIM <- as.factor(mydata3$DETCRIM)
mydata3$NOPRIOR <- as.factor(mydata3$NOPRIOR)
mydata3$SUB1 <- as.factor(mydata3$SUB1)
mydata3$ROUTE1 <- as.factor(mydata3$ROUTE1)
mydata3$FREQ1 <- as.factor(mydata3$FREQ1)
mydata3$FRSTUSE1 <- as.factor(mydata3$FRSTUSE1)
mydata3$SUB2 <- as.factor(mydata3$SUB2)
#mydata3$ROUTE2 <- as.factor(mydata3$ROUTE2)
#mydata3$FREQ2 <- as.factor(mydata3$FREQ2)
#mydata3$FRSTUSE2 <- as.factor(mydata3$FRSTUSE2)
mydata3$SUB3 <- as.factor(mydata3$SUB3)
#mydata3$ROUTE3 <- as.factor(mydata3$ROUTE3)
#mydata3$FREQ3 <- as.factor(mydata3$FREQ3)
#mydata3$FRSTUSE3 <- as.factor(mydata3$FRSTUSE3)
mydata3$NUMSUBS <- as.factor(mydata3$NUMSUBS)
mydata3$IDU <- as.factor(mydata3$IDU)
mydata3$ALCFLG <- as.factor(mydata3$ALCFLG)
mydata3$COKEFLG <- as.factor(mydata3$COKEFLG)
mydata3$MARFLG <- as.factor(mydata3$MARFLG)
mydata3$HERFLG <- as.factor(mydata3$HERFLG)
mydata3$METHFLG <- as.factor(mydata3$METHFLG)
mydata3$OPSYNFLG <- as.factor(mydata3$OPSYNFLG)
mydata3$PCPFLG <- as.factor(mydata3$PCPFLG)
mydata3$HALLFLG <- as.factor(mydata3$HALLFLG)
mydata3$MTHAMFLG <- as.factor(mydata3$MTHAMFLG)
mydata3$AMPHFLG <- as.factor(mydata3$AMPHFLG)
mydata3$STIMFLG <- as.factor(mydata3$STIMFLG)
mydata3$BENZFLG <- as.factor(mydata3$BENZFLG)
mydata3$TRNQFLG <- as.factor(mydata3$TRNQFLG)
mydata3$BARBFLG <- as.factor(mydata3$BARBFLG)
mydata3$SEDHPFLG <- as.factor(mydata3$SEDHPFLG)
mydata3$INHFLG <- as.factor(mydata3$INHFLG)
mydata3$OTCFLG <- as.factor(mydata3$OTCFLG)
mydata3$OTHERFLG <- as.factor(mydata3$OTHERFLG)
mydata3$ALCDRUG <- as.factor(mydata3$ALCDRUG)
#mydata3$DSMCRIT <- as.factor(mydata3$DSMCRIT)
mydata3$PSYPROB <- as.factor(mydata3$PSYPROB)
#mydata3$HLTHINS <- as.factor(mydata3$HLTHINS)
#mydata3$PRIMPAY <- as.factor(mydata3$PRIMPAY)

levels(mydata3)<-levels(mydata.train)

str(mydata.train)
str(mydata3)

predicted_values <- predict(rf, mydata3) # Use the rf classifier to make the predictions

write.csv(predicted_values,file="LOS_RandForest")

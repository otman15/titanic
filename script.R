

#our goal is to make prediction of survival of 
#passengers of the famous titanic ship


#load libraries
library(tidyverse) 
library(caret) 
library(rpart)
library(matrixStats)
library(e1071)
library(readr)
library(gbm)
library(class)


#II. Exploratory data analysis

#1. inspect data
train <- read.csv("raw_data/train.csv")
test <- read.csv("raw_data/test.csv")
head(train)
tail(train)
dim(train)

#rates 

train %>% summarise(avg_surv = mean(Survived))

#Sex :
#conc : Most of the survivors were female;
#Most of the females survived.


# plot - survival filled by sex with position_dodge
train %>%
  ggplot(aes(Survived, fill = Sex)) +
  geom_bar(position = position_dodge( ))

#plot - sex filled by survival
train %>% mutate(Survived = as.factor(Survived)) %>%
  ggplot(aes(Sex, fill = Survived)) + geom_bar()


#Age
#0:8 most likly to survive, most number of death 30:40,
#70:80 heighest proportion of death

train %>% ggplot(aes(Age)) + geom_density()

#Age by sex 
train %>%
  ggplot(aes(Age, y = ..count.., fill = Sex)) +
  geom_density(alpha = 0.2, position = "stack")

params <- train %>%
  summarize(mean = mean(Age), sd = sd(Age))

#The age is approximately normally distributed
train %>%
  ggplot(aes(sample = Age)) +
  geom_qq(dparams = params) +
  geom_abline()


#plot age density by survival

clean_data %>%
  ggplot(aes(Age, y = ..count.., fill = Survived)) +
  geom_density(alpha = 0.2)


#Survival by Fare :
#survived generally paid heigher fare
train  %>% mutate(Survived = as.factor(Survived)) %>%
  ggplot(aes(Survived, Fare)) +
  geom_boxplot() +
  scale_y_continuous(trans = "log2") +
  geom_jitter(alpha = 0.2)

#Pclass
#The majority of those who did not 
#survive were from third class.
#Most passengers in first class survived.
#Most passengers in other classes did not survive.

#Survival by class
train %>% mutate(Pclass = as.factor(Pclass), 
          Survived = as.factor(Survived)) %>%
  ggplot(aes(Survived, fill = Pclass)) +
  geom_bar(position = position_fill()) +
  ylab("Proportion")


#Survival by Age, Sex and Passenger Class
train %>% mutate(Survived = as.factor(Survived), 
        Pclass = as.factor(Pclass)) %>%
  ggplot(aes(Age, y = ..count.., fill = Survived)) +
  geom_density(position = "stack") +
  facet_grid(Sex ~ Pclass)

#Survival by Embarked
train %>% mutate(Survived = as.factor(Survived)) %>%
  ggplot(aes(Embarked, fill = Survived)) +
  geom_bar() + ylab("Proportion")
#Survival by Family_size
train %>% mutate(Survived = as.factor(Survived),
            family_size = (SibSp + Parch)) %>%
  ggplot(aes(family_size, fill = Survived)) +
  geom_bar()  + ylab("Proportion")

#Survival by cabin : most na >> drop this feature
cab <- str_extract(train$Cabin,"[A-Z]")

train %>% mutate(Survived = as.factor(Survived),
                 cab = as.factor(cab)) %>%
  ggplot(aes(cab, fill = Survived)) +
  geom_bar()  + ylab("Proportion")

#survival by title:

title <- gsub('(.*, )|(\\..*)', '', train$Name)
title <- case_when(
  title %in% c("Miss", "Mlle", "Lady", "Ms") ~ "Miss",
  title %in% c("Mme", "Mrs") ~ "Mrs",
  title == "Mr" ~ "Mr",
  title %in% c("Capt", "Col", "Don", "Dr", 
        "Jonkheer", "Major", "Rev", "Sir",
        "the Countess") ~ "Officer",
  title == "Master" ~ "Master")

train %>% mutate(title = as.factor(title), 
        Survived = as.factor(Survived)) %>%
  ggplot(aes(title, fill = Survived)) +
  geom_bar(position = "fill") +
  ylab("Frequency")



#II. preprocessing

#clean and tidy data

#nas : drop cabin 80% nas
Na_s <- map_df(train, function(v){
  total <- sum(is.na(v))
  proportion <- total/length(v)
  data.frame(total, proportion)
}) %>% data.frame(feature = names(train)) %>% 
  arrange(desc(total)) %>% 
  select(feature, total, proportion) %>%
  filter(total != 0)
Na_s
train$Embarked %>% table() # replace nas with "S"

#make the trainig data 

titanic <- train %>% 
  mutate(Survived = as.factor(Survived), 
         Sex = as.factor(Sex), 
         Pclass = factor(Pclass, ordered = TRUE), 
         Family_size = SibSp + Parch, 
         Age = ifelse(is.na(Age), 
                      median(Age, na.rm = TRUE), Age), 
         Embarked = as.factor(ifelse(is.na(Embarked), "S", 
                        Embarked)),
         Title = title) %>%
  select(Survived, Pclass, Sex, Age, Family_size, Fare,
         Embarked, Title)
#train set test set 
set.seed(1, sample.kind = "Rounding")

test_index <- createDataPartition(titanic$Survived, 
            times =1, p = 0.3, list = FALSE)
test_set <- titanic %>% slice(test_index)
train_set <- titanic %>% slice(-test_index)

#prepare final test data for submission :

title_test <- gsub('(.*, )|(\\..*)', '', test$Name)
title_test <- case_when(
  title_test %in% c("Miss", "Ms") ~ "Miss",
  title_test %in% c("Mrs", "Dona") ~ "Mrs",
  title_test == "Mr" ~ "Mr",
  title_test %in% c("Col", "Dr", "Rev") ~ "Officer",
  title_test == "Master" ~ "Master")

final_test <- test %>% 
  mutate(Sex = as.factor(Sex), 
        Pclass = factor(Pclass, ordered = TRUE), 
        Family_size = SibSp + Parch, 
        Age = ifelse(is.na(Age), 
                    median(Age, na.rm = TRUE), Age), 
        Embarked =  as.factor(ifelse(is.na(Embarked), "S", 
                       Embarked)),
        Title = title_test,
        Fare = ifelse(is.na(Fare),
         median(Fare, na.rm = TRUE), Fare)) %>%
  select(Pclass, Sex, Age, Family_size, Fare,
         Embarked, Title)

 # III models 

#1. baseline always predict die 
 set.seed(1, sample.kind = "Rounding")
base_pred <- ifelse(test_set$Survived == "0",
                        "0", "0") %>% as.factor()
conf_base <- confusionMatrix(base_pred,
                             test_set$Survived)

results <- data.frame(method = "base", accuracy =
                  conf_base$overall[["Accuracy"]])
results


#glm 

set.seed(1, sample.kind = "Rounding")

train_glm <- train(Survived~.,
                   method = "glm", data = train_set[, -8] )
pred_glm <- predict(train_glm, newdata = test_set[, -1])
conf_glm <- confusionMatrix(data = pred_glm,
                            reference = test_set$Survived)

results <- rbind(results, data.frame(method = "glm", 
                    accuracy = conf_glm$overall[["Accuracy"]]))
results

#knn 

set.seed(1, sample.kind = "Rounding")
train_knn <- train(Survived~., method = "knn", 
                   data = train_set,
            tuneGrid = data.frame(k = seq(3, 51, 2)))
pred_knn <- predict(train_knn, newdata = test_set[, -1])
conf_knn <- confusionMatrix(data = pred_knn, 
                            reference = test_set$Survived)
results <- rbind(results, data.frame(method = "knn", 
                      accuracy = conf_knn$overall[["Accuracy"]]))
results


# Classification tree
set.seed(1, sample.kind = "Rounding")

train_rpart <- train(Survived~., method = "rpart", 
                     data = train_set,
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)))
pred_rpart <- predict(train_rpart, newdata = test_set[, -1])
conf_rpart <- confusionMatrix(data = pred_rpart,
                              reference = test_set$Survived)

results <- rbind(results, data.frame(method = "rpart",
           accuracy = conf_rpart$overall[["Accuracy"]]))

results



#rf
trctrl <- trainControl(method = "repeatedcv",
                       number = 10, repeats = 3)
set.seed(1, sample.kind = "Rounding")

train_rf <- train(Survived~., method = "rf", 
               data = train_set,
               tuneGrid = data.frame(mtry =
                             seq(1:15)), ntree = 200,
               trControl=trctrl)
pred_rf <- predict(train_rf, newdata = test_set[, -1])
conf_rf <- confusionMatrix(data = pred_rf, 
                 reference = test_set$Survived)
results <- rbind(results, data.frame(method = "rf", 
             accuracy = conf_rf$overall[["Accuracy"]]))
results

#svm
trctrl <- trainControl(method = "repeatedcv",
                       number = 10, repeats = 3)
set.seed(1, sample.kind = "Rounding")

train_svm <- train(Survived~., method = "svmLinear", 
                  data = train_set, 
                  trControl=trctrl,
                  preProcess = c("center", "scale"),
                  tuneLength = 1)
pred_svm <- predict(train_svm, newdata = test_set[, -1])
conf_svm <- confusionMatrix(data = pred_svm, 
                     reference = test_set$Survived)
results <- rbind(results, data.frame(method = "svm", 
          accuracy = conf_svm$overall[["Accuracy"]]))
results

#gbm
 
fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE)

set.seed(1, sample.kind = "Rounding")

train_gbm <- train(Survived~., method = "gbm", 
                  data = train_set,
                  trControl = fitControl,
                  verbose=FALSE)
pred_gbm <- predict(train_gbm, newdata = test_set[, -1])
conf_gbm <- confusionMatrix(data = pred_gbm, 
                 reference = test_set$Survived)
results <- rbind(results, data.frame(method = "gbm", 
          accuracy = conf_gbm$overall[["Accuracy"]]))
results

# vote 
vote <- data.frame( pred_gbm = pred_gbm,
                  pred_rf = pred_rf, 
                  pred_svm = pred_svm)
vote <-vote %>% mutate_all(as.character) %>%
  mutate_all(as.numeric)
p_hat <- vote %>% as.matrix() %>% rowMeans()
y_hat <- ifelse(p_hat > 0.5, 1, 0) %>% as.factor()

conf_vote <- confusionMatrix(data = y_hat,
             reference = as.factor(test_set$Survived))

results <- rbind(results, data.frame(method = "vote",
           accuracy = conf_vote$overall[["Accuracy"]]))
results

#final model 
#we ll use svm rf and gbm to make a vote
#rf
trctrl <- trainControl(method = "repeatedcv",
                       number = 10, repeats = 3)
set.seed(1, sample.kind = "Rounding")

train_rf <- train(Survived~., method = "rf", 
                  data = titanic,
                  tuneGrid = data.frame(mtry =
                      seq(1:15)), ntree = 200,
                  trControl=trctrl)
pred_rf <- predict(train_rf, newdata = final_test)

#svm 
trctrl <- trainControl(method = "repeatedcv",
                       number = 10, repeats = 3)
set.seed(1, sample.kind = "Rounding")

train_svm <- train(Survived~., method = "svmLinear", 
                   data = titanic, 
                   trControl=trctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 1)
pred_svm <- predict(train_svm,
                    newdata = final_test)
#gbm

fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE)

set.seed(1, sample.kind = "Rounding")

train_gbm <- train(Survived~., method = "gbm", 
                   data = titanic,
                   trControl = fitControl,
                   verbose=FALSE)
pred_gbm <- predict(train_gbm, newdata = final_test)

# vote 

vote <- data.frame( pred_gbm = pred_gbm,
                    pred_rf = pred_rf, 
                    pred_svm = pred_svm)
vote <-vote %>% mutate_all(as.character) %>%
  mutate_all(as.numeric)
p_hat <- vote %>% as.matrix() %>% rowMeans()
predictions <- ifelse(p_hat > 0.5, 1, 0) %>% as.factor()

#submission

submission <- data.frame(PassengerId = test$PassengerId,
            Survived = as.numeric(as.character(predictions)))

write.csv(submission,file="submission.csv",row.names = F)


submission_rf <- data.frame(PassengerId = test$PassengerId,
                            Survived = as.numeric(as.character(pred_rf)))

write.csv(submission_rf,file="submission_rf.csv",row.names = F)












setwd("D:/projects/OnlineCourses/08 Practical Machine Learning/project")
training <- read.csv("pml-training.csv")
summary(training)
table(training["classe"])

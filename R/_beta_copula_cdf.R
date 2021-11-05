library(readr)
library(kdecopula)

u_train <- read_csv("R/_u_train_bicop.csv", col_names=FALSE)
u_test <- read_csv("R/_u_test_bicop.csv", col_names=FALSE)

u_train <- data.matrix(u_train)
u_test <- data.matrix(u_test)


cop_beta <- kdecop(u_train, method="beta")
cdf_beta <- pkdecop(u_test, cop_beta)
res_beta <- data.frame(beta=cdf_beta)
write_csv(res_beta, file="R/_cdf_beta.csv", col_names = FALSE, append=FALSE)
library('minpack.lm')

dat <- read.csv('SR_Data_2024Assessment.csv', sep = ';')

x <- dat$SSB[1:(length(dat$SSB)-1)]
y <- dat$Recruits_1[2:(length(dat$SSB)-0)]

m1 <- nlsLM(log(y) ~ (log_ac + log(x)) - log(1 + exp(log_bc + log(x))), control = list(maxiter = Inf))

print(exp(coef(m1)))

x <- dat$SSB[1:(length(dat$SSB))]
y <- dat$Recruits_0[1:(length(dat$SSB))]

m2 <- nlsLM(log(y) ~ (log_ac + log(x)) - log(1 + exp(log_bc + log(x))), control = list(maxiter = Inf))

print(exp(coef(m2)))





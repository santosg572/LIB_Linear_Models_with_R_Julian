library(faraway)

data (pima)

pima$diastolic [pima$diastolic == 0] <- NA
pima$glucose [pima$glucose == 0] <- NA
pima$triceps [pima$triceps == 0] <- NA
pima$insulin [pima$insulin == 0] <- NA
pima$bmi [pima$bmi == 0] <- NA

pima$test <- factor(pima$test)

print(summary (pima$test))

#hist(pima$diastolic)

#plot (density (pima$diastolic, na.rm=TRUE) )

plot (sort (pima$diastolic), pch=".")










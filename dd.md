
# LIB: Linear Models with R_Julian

## Preface

## C01 Int

### 1.1 Before tou start

### 1.2 Initial Data Analysis



> library(faraway)
> data (pima)
> pima

> summary(pima)

> sort(pima$diasrolic)


> pima$diastolic [pima$diastolic = = 0] < - NA
> pima$glucose [pima$glucose == 0] < - NA
> pima$triceps [pima$triceps == 0] < - NA
> pima$insulin [pima$insulin == 0] < - NA
> pima$bmi [pima$bmi == 0] < - NA

> pima$test < - factor (pima$test)
> summary (pima$test)

> plot (density (pima$diastolic, na . rm=TRUE) )

> plot (sort (pima$diastolic), pch=".")

> plot (diabetes ˜ diastolic,pima)
> plot (diabetes ˜ test, pima)

> pairs (pima)

### 1.3 When to Use Regression Analysis

### 1.4 History

* Rgression-type problems were first considered in the 18th century to aid navigation with the use of astronomy.
* Legendre developed the method of least squares in 1805. 
* Gauss claimed to have developed the method a few years earlier and in 1809 showed that least squares is the optimal solution when the errors are normally distributed. 
* The methodology was used almost exclusively in the physical sciences until later in the 19th century.
* Francis Galton coined the term regression to mediocrity in 1875 in reference to the simple regression equation in the form:

$$
\frac{y - \bar{y}}{SD_y} = r \frac{(x-\bar{x})}{SD_x}
$$

where r is the correlation between x and y.

* Galton used this equation to explain the
phenomenon that sons of tall fathers tend to be tall but not as tall as their fathers, while
sons of short fathers tend to be short but not as short as their fathers. This phenomenom is
called the regression effect. See Stigler (1986) for more of the history.


$$
y = rx
$$


> data (stat500)
> stat500 < - data.frame (scale (stat500))
> plot (final ˜ midterm, stat500)
> abline (0, l)

Now a student scoring, say 1 SD above
average on the midterm might reasonably expect to do equally well on the final.

> g < - lm (final ˜ midterm, stat500)
> abline (coef (g), lty=5)
> cor (stat500)


shallower

midterm

Correspondingly,

If exams managed to measure the ability of students perfectly

luck.

mediocrity.”

sophomore jinx



### Exercises

1. The dataset teengamb concerns a study of teenage gambling in Britain. Make a numerical and graphical summary of the data, commenting on any features that you find interesting. Limit the output you present to a quantity that a busy reader would find sufficient to get a basic understanding of the data.

2. The dataset uswages is drawn as a sample from the Current Population Survey in 1988. Make a numerical and graphical summary of the data as in the previous question.

3. The dataset prostate is from a study on 97 men with prostate cancer who were due to receive a radical prostatectomy. Make a numerical and graphical summary of the data as in the first question.


5. The dataset sat comes from a study entitled “Getting What You Pay For: The Debate Over Equity in Public School Expenditures.” Make a numerical and graphical summary of the data as in the first question.

5. The dataset divusa contains data on divorces in the United States from 1920 to 1996. Make a numerical and graphical summary of the data as in the first question.
 
## CHAPTER 2 Estimation


### 2.1 Linear Model

* Supongamos que necesitamos modelar la respuesta Y en terminos de tres predictores $X_1, X_2$ y $X_3$.
* Una forma muy general para el modelo podria ser>

$$
y = f(X_1, X_2, X_3) + \epsilon
$$

* $\epsilon$ es aditivo en este caso
* Un modelo seria el lineal

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \epsilon 
$$

$\beta_0$ es llamado el termino intercepto

* En un modelo lineal los parametros entran linealmente, los predictores no tienen que ser lineales por ejemplo.

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 \log X_2 + \beta_3 X_1X_2  + \epsilon  
$$

pero

$$
Y = \beta_0 + \beta_1 X_1^{\beta_2}  + \epsilon  
$$

### 2.2 Representacion Matricial

\begin{matrix}
Y_1 & X_{11}  & X_{12} & X_{13} \\ 
Y_2 & X_{21}  & X_{22} & X_{23} \\ 
... & ... & ... & ... \\ 
Y_n & X_{n1}  & X_{n2} & X_{n3} 
\end{matrix}


* n es el numero de obserrvaciones o casos

* Entonces podemos escribir el modelo como

$$
Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \beta_3 X_{3i} + \epsilon_i
$$

i=1,...,n

* Otra manera de escribir esta expresion es en forma matyricial:

$$
Y = X \beta + \epsilon
$$

donde

X = \begin{pmatrix}
1 & x_{11}  & x_{12} & x_{13} \\ 
1 & x_{21}  & x_{22} & x_{23} \\ 
... & ... & ... & ...\\
1 & x_{n1}  & x_{n2} & x_{n3}  
\end{pmatrix}

* Un ejemplo simple es el modelo nulo donde no hay predictor y justamente la media $y = \mu + \epsilon$

$$
\begin{pmatrix}
y_1 \\ 
y_2\\ 
... \\ 
y_n
\end{pmatrix} = \begin{pmatrix}
1\\ 
1\\ 
...\\ 
1
\end{pmatrix} \mu + \begin{pmatrix}
\epsilon_1\\ 
\epsilon_2\\ 
...\\ 
\epsilon_n
\end{pmatrix}
$$

### 2.3 Estinando $\beta$

* The regression model, $y = X \beta + \epsilon$, partitions the response into a systematic component $X\beta$
and a random component $\epsilon$
* We would like to choose $\beta$ so that the systematic part explains as much of the response as possible.
* The problem is to find $\beta$ so that $X\beta$ is as close to Y as possible. The best choice, the
estimate $\hat{\beta}$ , is apparent in the geometrical representation seen in Figure 2.1.
* $\hat{\beta}$ is, in this sense, the best estimate of $\beta$ within the model space. The response predicted
by the model is $\hat{y}=X\hat{\beta}$
or Hy where H is an orthogonal projection matrix.
* The difference between the actual response and the predicted response is denoted by and is called the
residuals.







### 2.4 Least Squares Estimation


* The estimation of $\beta$ can also be considered from a nongeometrical point of view. 
* We might define the best estimate of $\beta$ as the one which minimizes the sum of the squared
errors. 
* The least squares estimate of $\beta$, called $\hat{\beta}$ minimizes:


$$
\sum \epsilon^2 = \epsilon^T \epsilon = (y - X \beta)^T (y - X \beta)
$$

Differentiating with respect to $\beta$ and setting to zero, we find that $\hat{\beta}$
satisfies:

$$
X^T X \hat{\beta} = X^T y
$$

These are called the normal equations.

$$
\hat{\beta} = (X^T X)^{-1} X^T y \\
X \hat{\beta} = X (X^T X)^{-1} X^T y \\
\hat{y} = Hy
$$

$H=X(X^TX)^{-1}X^T$ is called the hat-matrix and is the orthogonal projection of y onto the
space spanned by X.

* The predicted or fitted values are $\hat{y} = Hy = X \hat{\beta}$
while the residuals are $\hat{\epsilon} = y - X \hat{\beta} = y = \hat{y} = (I - H)y$.

* The residual sum of squares (RSS) is $\hat{\epsilon}^T\hat{\epsilon} = y^T(I-H)^T (I-H)y = y^T(I-H)y$.





$$
\hat{\sigma}^2 = \frac{\hat{e}^T \hat{e}}{n-p} = \frac{RSS}{n-p}
$$


 ### 2.5 Examples of Calculating $\hat{\beta}$
 
 1. Cuando $y = \mu + \epsilon$, $X=\textbf{1}$  asi $X^T X = \textbf{1}^T \textbf{1} = n$ 
 $$
 \hat{\beta} = (X^T X)^{-1} X^T y = \frac{1}{n} 1^Ty = \bar{y}
 $$
 
 2. regresion lineal simple (un predictor):

$$
y_i = \beta_0 + \beta_1 x_i + \epsilon_i
$$

$$
\begin{pmatrix}
y_1\\ 
... \\
y_n 
\end{pmatrix} = \begin{pmatrix}
1 & x_1 \\ 
... & ...\\ 
1 & x_n 
\end{pmatrix} \begin{pmatrix}
\beta_0\\ 
\beta_1
\end{pmatrix} + \begin{pmatrix}
\epsilon_1\\
... \\
\epsilon_n
\end{pmatrix}  
$$

$$
y_i 
$$

$$
y_i = \overset{\beta'_0}{\overbrace{\beta_0 + \beta_1 \bar{x}}} +m 
$$

asi ahora:


### 2.6 Gauss-Markov Theorem


* $\hat{\beta}$ is a plausible estimator, but there are alternatives. Nonetheless, there are three
goodreasons to use least squares:

 1. It results from an orthogonal projection onto the model space. It makes sense
geometrically.
2. If the errors are independent and identically normally distributed, it is the maximum
likelihood estimator. Loosely put, the maximum likelihood estimate is the value of $\beta$
that maximizes the probability of the data that was observed.
3. The Gauss-Markov theorem states that $\hat{\beta}$ is the best linear unbiased estimate (BLUE).
 
 
 
 Nonetheless,
 
 goodreasons
 
 identically normally distributed
 
 the maximum likelihood estimator
 
 Loosely put
 
 the maximum likelihood estimate
 
 The Gauss-Markov theorem
 
 estimable function
 
 $\psi = c^T \beta$
 
 $Ea^T = c^T \beta \hspace{5mm}$   $\forall \beta$
 
 they are well worth considering

If X is of full rank

$c^T \hat{\beta} = \lambda^T X^T X \hat{\beta} = \lambda^T X^T y$

### 2.7 Goodness of Fit

* It is useful to have some measure of how well the model fits the data. One common
choice is $R^2$, the so-called coefficient of determination or percentage of variance explained:

$$
R^2 = 1 - \frac{\sum(\hat{y}_i - \bar{y}_i)^2}{\sum(y?i - \bar{y})} = 1 - \frac{RSS}{Total \ SS(Corrected \ for \ Mean)}
$$

An equivalent definition

$$
R^2 = \frac{\sum(\hat{y}_i - \bar{y})^2}{\sum(y_i - \bar{y})^2}
$$

### 2.8 Example

tortoise


**> data (gala) 
> gala
**

island

Endemics

highest elevation

adjacent

Nearest

I have filled in some missing values for simplicity



**

> x < - model.matrix ( ˜ Area + Elevation + Nearest + Scruz
+ Adjacent, gala)

> y < - gala$Species

> xtxi < - solve (t (x) %*% x)

> xtxi %*% t (x) %*% y

This is a very bad way to compute



> solve (crossprod (x, x), crossprod (x, y))

exacerbated

In the long run,




Thisted (1988).


### 2.9 Identifiability


$X^T X \hat{\beta} = X^T y$


$y_i = \mu + \alpha_i + \varepsilon_i$ i=1,2  j = 1,2,



Exercises
1. The dataset teengamb concerns a study of teenage gambling in Britain. Fit a regression
model with the expenditure on gambling as the response and the sex, status, income
and verbal score as predictors. Present the output.
(a) What percentage of variation in the response is explained by these predictors?
(b) Which observation has the largest (positive) residual? Give the case number.
(c) Compute the mean and median of the residuals.
(d) Compute the correlation of the residuals with the fitted values.
(e) Compute the correlation of the residuals with the income.
(f) For all other predictors held constant, what would be the difference in predicted
expenditure on gambling for a male compared to a female?
2. The dataset uswages is drawn as a sample from the Current Population Survey in
1988. Fit a model with weekly wages as the response and years of education and
experience as predictors. Report and give a simple interpretation to the regression

coefficient for years of education. Now fit the same model but with logged weekly
wages. Give an interpretation to the regression coefficient for years of education.
Which interpretation is more natural?
3. In this question, we investigate the relative merits of methods for computing the
coefficients. Generate some artificial data by:
Downloaded by [University of Toronto] at 16:20 23 May 2014
> x < - 1:20
> y < - x+rnorm(20)
Fit a polynomial in x for predicting y. Compute in two ways—by lm ( ) and by
using the direct calculation described in the chapter. At what degree of polynomial
does the direct calculation method fail? (Note the need for the I ( ) function in fitting
the polynomial, that is, lm(y˜x+I(x^2)) .
4. The dataset prostate comes from a study on 97 men with prostate cancer who were due
to receive a radical prostatectomy. Fit a model with lpsa as the response and l cavol as
the predictor. Record the residual standard error and the R2. Now add lweight, svi,
lpph, age, l cp, pgg45 and gleason to the model one at a time. For each model record
the residual standard error and the R2. Plot the trends in these two statistics.
5. Using the prostate data, plot lpsa against l cavol. Fit the regressions of lpsa on lcavol
and lcavol on lpsa. Display both regression lines on the plot. At what point do the two
lines intersect?



## CHAPTER 3 Inference (pp28)

* Until now, we have not found it necessary to assume any distributional form for the errors $\varepsilon$. 

* However, if we want to make any confidence intervals or perform any hypothesis tests, we will need to do this. 

* We have already assumed that the errors are independent and identically distributed (i.i.d.) with mean 0 and variance $\sigma^2$, so we have $\varepsilon ~ N(0, \sigma^2I)$. Now since $y=X\beta + \varepsilon$, we have $y ~ N(X\beta, \sigma^2I)$. which is a compact description of the regression model. 
* From this we find, using the fact that linear combinations of normally distributed values are also normal, that:

$$
\hat{\beta} = (X^TX)^{-1}X^T y ~ N(\beta, (X^TX)^{-1} \sigma^2)
$$

## 3.1 Hypothesis Tests to Compare Models

Given several predictors for a response, we might wonder whether all are needed.

* We will take $w$ represent the null hypothesis and $\Omega$ to represent the alternative.

* $RSS_w - RSS_{\Omega}$ is small, then the fit of the smaller model is almost as good as the largermodel and so we would prefer the smaller model on the grounds of simplicity

* This suggests that something like:


$$
\frac{RSS_w - RSS_{\Omega}}{RSS_{\Omega}}
$$

would be a potentially good test statistic where thedenominator is used for scaling purposes.


* As it happens, the same test statistic arises from the likelihood-ratio testing approach.We give an outline of the development: If $L(\beta, \sigma y)$ is the likelihood function, then the likelihood-ratio statistic is:

$$
\frac{max_{\beta, \sigma \in \Omega} L(\beta, \sigma y)}{max_{\beta, \sigma \in } L(\beta, \sigma y)}
$$


##  3.2 Testing Examples

Test of all the predictors

* Are any of the predictors useful in predicting the response? Let the full model ($\Omega$) be $y= X\beta + \varepsilon$ where X is a full-rank $n \times p$ matrix and the reduced model (w) be $y= \mu \varepsilon$. We would estimate $\mu$) by $\bar{y}$ . We write the null hypothesis as:

$$
H_0: \beta_1 = ... = \beta_{p-1} = 0
$$

> data(savings)
> savings


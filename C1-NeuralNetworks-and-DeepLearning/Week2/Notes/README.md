# Neural Networks Basics
Set up a machine learning problem with a neural network mindset and use vectorization to speed up your models.

**Learning Objectives**
- Build a logistic regression model structured as a shallow neural network
- Build the general architecture of a learning algorithm, including parameter initialization, cost function and gradient calculation, and optimization implemetation (gradient descent)
- Implement computationally efficient and highly vectorized versions of models
- Compute derivatives for logistic regression, using a backpropagation mindset
- Use Numpy functions and Numpy matrix/vector operations
- Work with iPython Notebooks
- Implement vectorization across multiple training examples
- Explain the concept of broadcasting

## Binary Classification
Logistic regression is an algorithm for binary classification.

Example of binary classification: input an image and want to output a label to recognize this image as either being a cat (output 1) or not cat (output 0). Use `y` to denote output label.

The computer stores an image by using 3 separate matrices corresponding to the red, green and blue color channels of this image.

Image 64x62 pixels -> 364x64 matrices RGB

Define a feature vector `x` corresponding to the image.

_You have a picture of a cat. To help a computer recognize this cat, we need to break down the image into numbers that represent its features, such as colors, shapes, and textures. These numbers are organized into a list, which we call a feature vector._

Total dimension of vector `x` is 

$64*64*3 = 12288$

Use $n = n_x = 12288$ to represent the dimension of input feature `x`.

In binary classification, the goal is to learn a classifier that can input an image represented by this feature vector x, then predict whether the corresponding label `y` is 1 or 0. 

![alt text](_assets/BinaryCls.png)

### Notation
Training example is represented by a pare `(x,y)`, where `x` is an x-dimensional feature vector and `y` is the label, either 0 or 1.

Training set will comprise `m` training examples: 
$(x^{(1)},y^{(1)}), (x^{(2)},y^{(2)})...(x^{(m)},y^{(m)})$

To emphasize this is a number of training example, we use $M=M_{train}$

Test set: $M_{test}$ test examples

To output all of the training examples into a more compact notation, we define a matrix, capital `X`. This matrix `X` has `m` columns (where `m` is the number of training examples), and the number of row is $n_X$

![alt text](_assets/matrixX.png)

$X \in {\mathbb{R}}^{n * m}$

In Python, find a shape of a matrix

```python
X.shape = (n, m)
```

For output label `Y`, we stack `Y` in columns

$Y=[y^{(1)} y^{(2)} ... y^{(m)}]$

$Y \in {\mathbb{R}}^{1 * m}$

```python
Y.shape = (1,m)
```

_Logistic regression is a statistical method used to predict the outcome of a binary classification problem, which means it helps us decide between two possible outcomes. For example, imagine you want to determine whether an email is spam (1) or not spam (0). Logistic regression takes various features of the email, such as the presence of certain words or the length of the message, and combines them to produce a probability score. This score tells us how likely it is that the email belongs to one of the two categories._

_To visualize this, think of logistic regression as a seesaw. On one side, you have all the features of the email, and on the other side, you have the two outcomes: spam and not spam. The logistic regression model calculates a balance point (the probability) that helps us decide which side the seesaw tips towards. If the probability is above a certain threshold (like 0.5), we classify the email as spam; if it's below, we classify it as not spam. This way, logistic regression provides a clear and interpretable method for making predictions based on input data._

![alt text](_assets/Notation.png)

## Logistic Regression

This is a learning algorithm that you use when the output labels `y in a supervised learning problem are all either 0 or 1, so for binary classification problems.

Given an input feature vector `x`, you want an algorithm that can output a prediction, which is $\hat{y}$, which is your estimate of Y. So you want $\hat{y}$ to be the probability of the chance that, `y` is equal to 1 given the input features `x`. I.e. if x is a picture, you want $\hat{y}$ to tell you, what is the chance that this is a cat picture ?

Given `x`, wnat $\hat{y} = P(y=1|x)$

`x` is an $n_x$ dimensional vector $x \in {\mathbb{R}}^{n_x}$, given that the parameters of logistic regression will be `w` which is also an $n_x$ dimensional vector $w \in {\mathbb{R}}^{n_x}$, together with `b` which is just a real number $b \in {\mathbb{R}}$.

Output $\hat{y} = \sigma(w^T*x+b)$, denote $z=w^T*x+b$

![alt text](_assets/LROutput.png)

Sigmoid function: $\sigma(z) = 1/(1+e^{-z})$

![alt text](_assets/LROutput2.png)

* If `z` is very large, then `e` to the negative `z` will be close to 0. So then sigmoid of `z` will be approximately 1 over 1 plus something very close to 0, because `e` to the negative of very large number will be close to 0. So this is close to 1.
* If `z` is very small, or it's a very large negative number, then signoid of `z` becomes 1 over 1 plus `e` to the negative `z`, and this becomes a huge number. So this becomes, think of it as 1 over 1 plus a number that is very, very big, and so, that's close to 0. 

![alt text](_assets/sigmoidProperties.png)

So when you implement logistic regression, your job is to try to learn parameters `w` and `b` so that $\hat{y}$ becomes a good estimate of the chance of `y` being equal to 1. 

When we program neural networks, we usually keep the parameter `w` and `b` separate.

![alt text](_assets/Q1.png)

## Logistic Regression Cost Function
To train the parameters `w` and `b` of the logistic regression model, you need to define a cost function.

$\hat{y} = \sigma(w^T*x+b)$

denote $z=w^T*x+b$

$\sigma(z) = {1 \over (1+e^{-z})}$

Given a training set of `m` training examples $(x^{(1)},y^{(1)}), (x^{(2)},y^{(2)})...(x^{(m)},y^{(m)})$, we want to find parameters `w` and `b` so that at least on the training set, the outputs you have, the predictions you have on the training set ${\hat{y}}^{(i)} \approx y^{(i)}$.

And of course for each training example, we're using these superscripts with round brackets with parentheses to index into different train examples.

Prediction on the training example `i` (i-th training example)

$\hat{y}^{(i)} = \sigma(w^T*x^{(i)}+b)$

denote $z^{(i)}=w^T*x^{(i)}+b$

$\sigma(z) = {1 \over (1+e^{-z^{(i)}})}$

**Loss function** is a way to measure how well your model is performing. (difference between a predicted label and a true label) (how good our output $\hat{y}$ is when the true label is y)

$\ell(\hat{y}, y) = {{1} \over {2}} {(\hat{y} - y)}^2$

In logistic regression, people don't usually do this because when you come to learn the parameters, you find that the optimization problem becomes non-convex, which is optimization problem with multiple local optima. So gradient descent may not find a global optimum (squared error seems like it might be a reasonable choice except that it makes gradient descent not work well). So in logistic regression, we define another loss function that plays a similar role as squared error but will give us an optimization problem that is convex. 

![alt text](_assets/costfunction1.png)

**Loss function for logistic regression.**

$\ell(\hat{y}, y) = -(ylog(\hat{y}) + (1-y)log(1-\hat{y}))$

* If y=1: then $\ell(\hat{y}, y) = - log(\hat{y})$, you want $- log(\hat{y})$ to be as small as possible, so you want $log(\hat{y})$ to be as large as possible, that means you want $\hat{y}$ to be large but $\hat{y}$ cannot be bigger than 1.
* If y=0: then $\ell(\hat{y}, y) = - log(1-\hat{y})$, you want $- log(1-\hat{y})$ to be as small as possible, so you want $log(1-\hat{y})$ to be as large as possible, that means you want $1-\hat{y}$ to be large -> $\hat{y}$ to be small, but it cannot smaller than 0.

**The loss function was defined with respect to a single training example. It measures how well you're doing on a single training example.**

**Cost function**: measure how well you are doing in the entire training set.

$J(w, b) =  {1 \over m} \Sigma_{i=1}^m \ell(\hat{y}^{(i)}, y^{(i)})$ 

$\hat{y}^{(i)}$ is of course the prediction output by your logistic regression algorithm using, you know, a particular set of parameters `w` and `b`.

$J(w, b) =  -{1 \over m} \Sigma_{i=1}^m (ylog(\hat{y}) + (1-y)log(1-\hat{y}))$

Loss function is applied to just a single training example. 

The cost function is the cost of your parameters, so in training your logistic regression model, we're going to try to find parameters `w` and `b` that minimize the overall cost function `J`. 

![alt text](_assets/costfunction2.png)

Logistic regression can be viewed as a very, very small neural network.

![alt text](_assets/sigmoidProperties.png)

## Gradient Descent
Recap

$\hat{y} = \sigma(w^T*x+b)$

denote $z=w^T*x+b$

$\sigma(z) = {1 \over (1+e^{-z})}$

$J(w, b) = {1 \over m} \Sigma_{i=1}^m \ell(\hat{y}^{(i)}, y^{(i)}) = -{1 \over m} \Sigma_{i=1}^m (ylog(\hat{y}) + (1-y)log(1-\hat{y}))$

Want to find w, b that minimize J(w,b)

![alt text](_assets/costGraph.png)

And what we want to do really is to find the value of w and b that corresponds to the minimum of the cost function J. 

It turns out that this particular cost function J is a convex function. 

So it's just a single big bowl, so this is a convex function.

![alt text](_assets/convex.png)

This is as opposed to functions that look like this, which are non convex and has lots of different local optimal.

![alt text](_assets/nonconvex.png)

So the fact that our cost function J of w, b (J(w,b)) as defined here is convex, is one of the huge reasons why we use this particular cost function J for logistic regression.


So to find a good value for the parameters, what we'll do is initialize w and b to some initial value may be denoted by that little red dot. And for logistic regression, almost any initialization method works. Usually you Initialize the values of 0. 

Random initialization also works, but people don't usually do that for logistic regression. But because this function is convex, no matter where you initialize, you should get to the same point or roughly the same point.

And what gradient descent does is it starts at that initial point and then takes a step in the steepest downhill direction. So after one step of gradient descent, you might end up there because it's trying to take a step downhill in the direction of steepest descent or as quickly down as possible. So that's one iteration of gradient descent. And after iterations of gradient descent, you might stop there, three iterations and so on. Until, hopefully, you are converged to the global optimum.

![alt text](_assets/gradientdescent.png)

Let's say that there's some function J of w (J(w)) that you want to minimize and maybe that function looks like this to make this easier to draw. I'm going to ignore b for now just to make this one dimensional plot instead of a higher dimensional plot.

Gradient deseng process

Repeate {
    $w := w - \alpha {dJ(w) \over dw}$
}

This is repeated until the algorithm converges.

$\alpha$ here is **learning rate**, and controls how big a step we take on each iteration are gradient descent.

$dJ(w) \over dw$: this quantity here, this is a derivative. This is basically the update of the change you want to make to the parameters `w`.

In code, we use dw to represent this derivative term

$w:= w - \alpha dw$

Remember that derivative is the slope of the function at a point. The slope of the function is the height devided by the width right of the lower triangle.

* When the derivative is positive, you take a step to the left, so the gradient descent makes the algorithm slowly decrease the w parameter.
* When the derivative is negative, you take a step to the left, so the gradient descent makes the algorithm slowly increase the w parameter.

![alt text](_assets/derivativeGraph.png)

The overall intuition for now is that this term represents the slope of the function and we want to know the slope of the function at the current setting of the parameters so that we can take these steps of steepest descent so that we know what direction to step in in order to go downhill on the cost function J.

In term of cost function J(w,b). updates are as follow

* $w := w - \alpha {dJ(w,b) \over dw}$
* $b := b - \alpha {dJ(w,b) \over db}$

![alt text](_assets/gradientdescent2.png)

Notation

![alt text](_assets/derivativeNotation.png)

_I don't think it's super important that you understand calculus but in case you see this, I want to make sure that you don't think too much of this. Which is that in calculus this term here we actually write as follows, that funny squiggle symbol. So this symbol, this is actually just the lower case `d` in a fancy font, in a stylized font. But when you see this expression, all this means is this is the of J of w, b or really the slope of the function J of w, b how much that function slopes in the w direction. And the rule of the notation and calculus, which I think is in total logical. But the rule in the notation for calculus, which I think just makes things much more complicated than you need to be is that if J is a function of two or more variables, then instead of using lower case d. You use this funny symbol._

_This is called a partial derivative symbol, but don't worry about this. And if J is a function of only one variable, then you use lower case d. So the only difference between whether you use this funny partial derivative symbol or lower case d. As we did on top is whether J is a function of two or more variables. In which case use this symbol, the partial derivative symbol or J is only a function of one variable. Then you use lower case d. This is one of those funny rules of notation and calculus that I think just make things more complicated than they need to be._

_But if you see this partial derivative symbol, all it means is you're measuring the slope of the function with respect to one of the variables, and similarly to adhere to the, formally correct mathematical notation calculus because here J has two inputs. Not just one. This thing on the bottom should be written with this partial derivative simple, but it really means the same thing as, almost the same thing as lowercase d. Finally, when you implement this in code, we're going to use the convention that this quantity really the amount I wish you update w will denote as the variable d w in your code. And this quantity, right, the amount by which you want to update b with the note by the variable db in your code._

![alt text](_assets/Q3.png)


## Derivatives



# Practical Aspects of Deep Learning

**Learning Objectives**
* Give examples of how different types of initializations can lead to different results
* Examine the importance of initialization in complex neural networks
* Explain the difference between train/dev/test sets
* Diagnose the bias and variance issues in your model
* Assess the right time and place for using regularization methods such as dropout or L2 regularization
* Explain Vanishing and Exploding gradients and how to deal with them
* Use gradient checking to verify the accuracy of your backpropagation implementation
* Apply zeros initialization, random initialization, and He initialization
* Apply regularization to a deep learning model

- [Practical Aspects of Deep Learning](#practical-aspects-of-deep-learning)
  - [Train / Dev / Test sets](#train--dev--test-sets)
  - [Bias / Variance](#bias--variance)
  - [Basic Recipe for Machine Learning](#basic-recipe-for-machine-learning)
  - [Regularization](#regularization)
  - [Why Regularization Reduces Overfitting?](#why-regularization-reduces-overfitting)
  - [Dropout Regularization](#dropout-regularization)
    - [Implement dropout ("Inverted dropout")](#implement-dropout-inverted-dropout)
    - [Making predictions at test time](#making-predictions-at-test-time)
      - [Why divide by keep\_prob? (the expectation argument)](#why-divide-by-keep_prob-the-expectation-argument)
  - [Understanding Dropout](#understanding-dropout)
  - [Other Regularization Methods](#other-regularization-methods)
  - [Normalizing Inputs](#normalizing-inputs)
  - [Vanishing / Exploding Gradients](#vanishing--exploding-gradients)
  - [Weight Initialization for Deep Networks](#weight-initialization-for-deep-networks)
  - [Numerical Approximation of Gradients](#numerical-approximation-of-gradients)
  - [Gradient Checking](#gradient-checking)
  - [Gradient Checking Implementation Notes](#gradient-checking-implementation-notes)

## Train / Dev / Test sets
When training NN, there is a lot of decisions to make:
* Number of layers
* Number of hidden units
* Learning rates
* What activation functions to use
* etc

Applied ML is a highly iterative process.

![alt text](_assets/MLProcess.png)

* Start with an idea (Build NN with certain # of layers, # of hidden units, dataset, etc...)
* Code it up and try it by running the code.
* Run an experiment and get back a result that can tell how well this particular network or this particular configuration works.
* Based on the outcome, refine the ideas, change the choices and keep iterating to try ti find a better NN.

DL has great success in: Natural Language Processing, Computer Vision, Speech Recognition, Structural data (ads, web search (shopping websites or any website that wants to deliver great search results when you enter terms into a search bar), computer security, logistic, etc...)...

-> Even very experienced deep learning people find it almost impossible to correctly guess the best choice of hyperparameters the very first time. And so today, applied deep learning is a very iterative process where you just have to go around this cycle many times to hopefully find a good choice of network for your application.

Data can be devided in 3 sets:
* Training set
* Hold-out cross validation set (development set)(dev set)
* Test set

![alt text](_assets/DataSplit.png)

1. Training algorithms on training set
2. Use dev set to see which of many different models performs best on dev set.
3. After having done the above 2 steps long enough, take the best model and evaluate it on test set.

Dev set needs to be big enough to evalute two different algorithm choices or ten different algorithm choices and quickly decide which one is doing better.

The main goal of your test set is, given your final classifier, to give you a pretty confident estimate of how well it's doing. 

With small dataset: 70/30/0 or 60/20/20

With big dataset:

Example: We have 1 000 000 examples, then only 10 000 can be used for dev set and 10 000 can be used for test set.

-> 98% training, 1% dev and 1% test

If we have even more than 1 000 000 examples, then

-> 99.5% training, 0.4% dev and 0.1% test

Let's say you're building an app that lets users upload a lot of pictures and your goal is to find pictures of cats in order to show your users. Maybe all your users are cat lovers. 

Mismatched train/test distribution
* Training set: Cat pictures from webpages
* Dev/test sets: Cat pictures from users using your app

Turns out a lot of webpages have very high resolution, very professional, very nicely framed pictures of cats. But maybe your users are uploading, you know, blurrier, lower res images just taken with a cell phone camera in a more casual condition. And so these two distributions of data may be different.

-> **Make sure dev and test sets come from the same distribution**

Because you will be using the dev set to evaluate a lot of different models and trying really hard to improve performance on the dev set, it's nice if your dev set comes from the same distribution as your test set.

Because deep learning algorithms have such a huge hunger for training data, one trend I'm seeing is that you might use all sorts of creative tactics, such as crawling webpages, in order to acquire a much bigger training set than you would otherwise have. Even if part of the cost of that is then that your training set data might not come from the same distribution as your dev and test sets. But you find that so long as you follow this rule of thumb, that progress in your machine learning algorithm will be faster.

Not having a test set might be okay. (Only dev set).

Remember, the goal of the test set is to give you a unbiased estimate of the performance of your final network, of the network that you selected. But if you don't need that unbiased estimate, then it might be okay to not have a test set. So what you do, if you have only a dev set but not a test set, is you train on the training set and then you try different model architectures. Evaluate them on the dev set, and then use that to iterate and try to get to a good model. Because you've fit your data to the dev set, this no longer gives you an unbiased estimate of performance. But if you don't need one, that might be perfectly fine.

In the machine learning world, when you have just a train and a dev set but no separate test set, most people will call this a training set and they will call the dev set the test set. But what they actually end up doing is using the test set as a hold-out cross validation set. Which maybe isn't completely a great use of terminology, because they're then overfitting to the test set. So when the team tells you that they have only a train and a test set, I would just be cautious and think, do they really have a train dev set? Because they're overfitting to the test set. Culturally, it might be difficult to change some of these team's terminology and get them to call it a trained dev set rather than a trained test set, even though I think calling it a train and development set would be more correct terminology. And this is actually okay practice if you don't need a completely unbiased estimate of the performance of your algorithm

## Bias / Variance

![alt text](_assets/highBias.png)

![alt text](_assets/highBiasLine.png)

Let's say you have a data set that looks like this. If you fit a straight line to the data, maybe you get a logistic regression fit to that. This is not a very good fit to the data, and so there's a cause of high bias. Or we say that this is underfitting the data. 

![alt text](_assets/highVariance.png)

![alt text](_assets/highVarianceLine.png)

On the opposite end, if you fit an incredibly complex classifier, maybe a deep neural network. Or a new network with a lot of hidden units, maybe you can fit the data perfectly. But that doesn't look like a great fit either. So this is a classifier with high variance, and this is overfitting the data.

![alt text](_assets/justRight.png)

![alt text](_assets/justRightLine.png)

And there might be some classifier in between with a medium level of complexity that maybe fits a curve like that. That looks like a much more reasonable fit to the data. So that's the, and call that just right somewhere in each tree.

So in a 2d example like this, with just two features, x1 and x2, you can plot the data and visualize bias and variance. In high dimensional problems, you can't plot the data and visualize the decision boundary.

![alt text](_assets/CatClassification.png)

The two key numbers to look at to understand bias and variance will be:
* The trading set error.
* The dev set, or the development set error.

So, for the sake of argument, let's say that recognizing cats in pictures is something that people can do nearly perfectly, right? And so let's say your trading size error is 1% and your dev set error is, for the sake of argument, let's say, is 11%. So in this example, you're doing very well on the training set, but you're doing relatively poorly on the development set. So this looks like you might have overfit the training set. That somehow you're not generalizing well to this holdout cost validation set to development set. And so if you have an example like this, we will say this has high variance. So by looking at the training set error and the development set error, you would be able to render a diagnosis of your algorithm having high variance. 

Now let's say that you measure your training set in your dev set error and you get a different result. Let's say that your training set error is 15%. I'm writing your training set error in the top row and your dev set error is 16%. In this case, assuming that humans achieve roughly 0% error, that humans can look at these pictures and just tell if it's cat or not. Then it looks like the algorithm is not even doing very well on the training set. So if it's not even fitting the training data, as seen that well, then this is underfitting the data. And so this algorithm has high bias. 

But in contrast, this is actually generalizing at a reasonable level to the dev set, whereas performance of the dev set is only 1% worse as performance on the training set. So this algorithm has a problem of high bias because it's not even training, it's not even fitting the training set well. This is similar to the leftmost plot we had on the previous slide. 

Now here's another example. Let's say that you have 15% training set error. So that's pretty high bias. But when you evaluate on a dev set, it does even worse, maybe it does 30%. In this case, I would diagnose this algorithm as having high bias because it's not doing that well on the training set and high variance. So this is really the worst of both worlds. 

And one last example, if you have 0.5 training set error and 1% dev set error. Then maybe our users are quite happy that you have a cat costly with only 1% error, then this would have low bias and low variance. One subtlety that I'll just briefly mention, but we'll leave to a later video to discuss in detail. 

One subtlety that I'll just briefly mention, but we'll leave to a later video to discuss in detail is that this analysis is predicated on the assumption that human level performance gets nearly 0% error. Or more generally they're the optimal error, sometimes called Bayes error for the so the bayesian optimal error is nearly 0%. 

![alt text](_assets/Bias-Variance.png)

But it turns out that if the optimal error or the Bayes error were much higher, say it were 15%. Then if you look at this classifier, 15% is actually perfectly reasonable for training set. And you wouldn't say it as high bias and also have pretty low variance. So the case of how to analyze bias and variance when no classifier can do very well. For example, if you have really blurry images so that even a human or just no system could possibly do very well. Then maybe Bayes error is much higher. And then there's some details of how this analysis will change. 

But leaving aside this subtlety for now, the takeaway is that by looking at your trading set error. You can get a sense of how well you're fitting at least the training data. And so that tells you if you have a bias problem. And then looking at how much higher your error goes when you go from the training set to the dev set. That should give you a sense of how bad is the variance problem. So are you doing a good job generalizing from the training set to the dev set that gives you a sense of your variance? All this is under the assumption that the Bayes error is quite small and that your train and your dev sets are drawn from the same distribution.

![alt text](_assets/highBiasHighVariance.png)

So you remember we said that a classifier like this, a linear classifier, has high bias because it under fits the data. So this would be a classifier that is mostly linear and therefore under fits the data. We'll join this in purple. But if somehow your classifier does some weird things, then it's actually overfitting parts of the data as well. So the classifier that I drew in purple has both high bias and high variance. There's high bias because by being a mostly linear classifier, it's just not fitting this quadratic light shape that well. But by having too much flexibility in the middle, it somehow gets this example. And this example overfits those two examples as well. So this classifier kind of has high bias because it was mostly linear, but you needed maybe a curve function, a quadratic function. 

And it has high variance because it had too much flexibility to fit those two mislabeled outlier examples in the middle as well. In case this seems contrived, well, it is. This example is a little bit contrived in two dimensions, but with very high dimensional inputs. You actually do get things with high bias in some regions and high variance in some regions. And so it is possible to get cross files like this in high dimensional inputs that seem less contrived.

* Bias — The model’s assumptions are too strong → can’t capture the real pattern. Error from wrong assumptions. Model is too simple. (Using a straight line to fit a curved trend.)
* Variance — The model changes too much when given different data → it’s too sensitive. Error from too much sensitivity to training data. Model is too complex. (Drawing a zigzag line through every single data point.)

* Bias wants to simplify (too much = underfit)
* Variance wants to memorize (too much = overfit)

Imagine we have data points that form a curved pattern, like this:
```
  ^
y |       *
  |    *      *
  |  *          *
  | *              *
  ------------------------>
                   x
```

1. High Bias, Low Variance — Underfitting

A simple straight line: y=ax + b

Behavior:
* The line is too simple to capture the curve.
* It misses the overall shape — makes big errors everywhere.
* But every time we retrain on different data, it still gives a similar straight line.

```
  ^
y |       *
  |    *      *
  |  *    ---line---*
  | *              *
  ------------------------>
```

Meaning:
* High bias: strong assumption (data must be linear).
* Low variance: small data changes don’t affect the line much.
* Result: always wrong in the same way.

Like using a ruler to draw a straight line through a banana 🍌 — it’ll never fit.

1. Low Bias, High Variance — Overfitting

A very flexible curve, e.g., a high-degree polynomial.

Behavior:
* The curve passes exactly through every data point.
* It learns even the noise in the data.
* When we use new data, the curve shape changes completely.
```
  ^
y |      *  
  |   *---*---*
  |  *   \_/   *
  | *__--     --__*
  ------------------------>
```

Meaning:
* Low bias: model flexible enough to learn almost anything.
* High variance: small data change causes big curve change.
* Result: great on training data, poor on test data.

Like memorizing exam questions — perfect on training, fails on new tests.

3. Low Bias, Low Variance — Good Fit (Ideal)

A moderately flexible function, maybe a quadratic:
* It captures the general curve well.
* Doesn’t follow every tiny noise point.
* If data changes a bit, curve stays mostly the same.

```
  ^
y |       *
  |    *      *
  |  *   ---curve---*
  | *              *
  ------------------------>
```

Meaning:
* Low bias: model can represent real trend.
* Low variance: stable results on new data.
* Result: generalizes well.

Like understanding concepts, not just memorizing examples.

4. High Bias, High Variance — Worst Case

Modle:
* Either model is too simple and unstable (e.g., bad hyperparameters).
* Or training data is too small or too noisy.

```
  ^
y |  *       *
  |   *   *
  |    line moves randomly
  ------------------------>
```

Meaning:
* High bias: can’t capture shape.
* High variance: changes drastically each time.
* Result: unpredictable and inaccurate.

|Type|	Bias|	Variance|	Behavior|	Solution|
|-|-|-|-|-|
|High Bias, Low Variance|	High|	Low|	Underfitting	|Add layers, features, train longer|
|Low Bias, High Variance|	Low	|High|	Overfitting|	Use more data, regularization, dropout|
|High Bias, High Variance|	High|	High|	Bad model|	Redesign model|
|Low Bias, Low Variance|	Low|	Low|	Ideal	|Great generalization|

## Basic Recipe for Machine Learning
After having trained in an initial model, I will first ask, does your algorithm have high bias? And so, to try and evaluate if there is high bias, you should look at, the training set or the training data performance. And so, if it does have high bias, does not even fitting in the training set that well, some things you could try would be to try pick a network:
* More hidden layers or more hidden units.
* Or you could train it longer, maybe run trains longer.
* Try some more advanced optimization algorithms.
* Maybe find a new NN architecture that's better suited for this problem (this maybe works, maybe not). Getting a bigger networks almost always helps. 
* Training longer doesn't always help, but it doesn't hurt. Keep doing this until we get rid of high bias problem.

Usually, if you have a big enough network, you should usually be able to fit the training data well, so long as it's a problem that is possible for someone to do, alright? If the image is very blurry, it may be impossible to fit it, but if at least a human can do well on the task, if you think Bayes error is not too high, then by training a big enough network you should be able to, hopefully, do well, at least on the training set, to at least fit or overfit the training set. 

Once you've reduce bias to acceptable amounts, I will then ask, do you have a variance problem? And so to evaluate that I would look at dev set performance. Are you able to generalize, from a pretty good training set performance, to having a pretty good dev set performance? 

And if you have high variance:
* Best way to solve a high variance problem is to get more data.
* But sometimes you can't get more data, you could try regularization, to try to reduce overfitting.
* If you can find a more appropriate neural network architecture, sometimes that can reduce your variance problem as well, as well as reduce your bias problem. 

When Andrew said:

“Changing the architecture of your model can help improve variance,”

He didn’t mean to make the network bigger or deeper — he meant adjusting the architecture in a way that reduces overfitting. So, he was referring to reducing variance by simplifying or regularizing the model.

That could mean:
* Making the network smaller (fewer layers or units),
* Adding dropout,
* Adding L2 regularization,
* Using batch normalization,
* Or using data augmentation.

I try these things and I kind of keep going back, until, hopefully, you find something with both low bias and low variance, whereupon you would be done. 

A couple of points to notice. 
* First, is that depending on whether you have high bias or high variance, the set of things you should try could be quite different.  I'll usually use the training dev set to try to diagnose if you have a bias or variance problem, and then use that to select the appropriate subset of things to try. For example, if you actually have a high bias problem, getting more training data is actually not going to help. Or, at least it's not the most efficient thing to do, alright? So being clear on how much of a bias problem or variance problem or both, can help you focus on selecting the most useful things to try. 
* Second, in the earlier era of machine learning, there used to be a lot of discussion on what is called the bias variance tradeoff. And the reason for that was that, for a lot of the things you could try, you could increase bias and reduce variance, or reduce bias and increase variance. But, back in the pre-deep learning era, we didn't have many tools, we didn't have as many tools that just reduce bias, or that just reduce variance without hurting the other one. But in the modern deep learning, big data era, so long as you can keep training a bigger network, and so long as you can keep getting more data, which isn't always the case for either of these, but if that's the case, then getting a bigger network almost always just reduces your bias, without necessarily hurting your variance, so long as you regularize appropriately. And getting more data, pretty much always reduces your variance and doesn't hurt your bias much. 

So what's really happened is that, with these two steps, the ability to train, pick a network, or get more data, we now have tools to drive down bias and just drive down bias, or drive down variance and just drive down variance, without really hurting the other thing that much. And I think this has been one of the big reasons that deep learning has been so useful for supervised learning, that there's much less of this tradeoff where you have to carefully balance bias and variance, but sometimes, you just have more options for reducing bias or reducing variance, without necessarily increasing the other one. 

And, in fact, so last, you have a well-regularized network. Training a bigger network almost never hurts. And the main cost of training a neural network that's too big is just computational time, so long as you're regularizing. 

![alt text](_assets/MLBasicRecipe.png)

## Regularization
If you suspect your neural network is overfitting your data, that is, you have a high variance problem, one of the first things you should try is probably regularization. The other way to address high variance is to get more training data that's also quite reliable. But you can't always get more training data, or it could be expensive to get more data. But adding regularization will often help to prevent overfitting, or to reduce variance in your network.

Recall that for logistic regression, you try to minimize the cost function J.

$J(w,b) = {1 \over m} \Sigma_{i=1}^m \ell(\hat{y}^{(i)}, y^{(i)})$ 

To add regularization to logistic regression, what you do is add to it, this thing, lambda, which is called the regularization parameter.

$J(w,b) = {1 \over m} \Sigma_{i=1}^m \ell(\hat{y}^{(i)}, y^{(i)}) + {\lambda \over 2m}||w||^2_2$ 

w is $n_x$ dimensional parameter vector and b is a real number

$||w||^2_2 = \Sigma_{i=1}^{n_x} w^2_j = w^Tw$

Square Euclidean norm of the prime to vector w.

This is called L2 regularization because here we are using the Euclidean norm, or L2 norm with the parameter vector w.

Why do you regularize just the parameter w? Why don't we add something here, you know, about b as well? In practice, you could do this, but I usually just omit this. Because if you look at your parameters, w is usually a pretty high dimensional parameter vector, especially with a high variance problem. Maybe w just has a lot of parameters, so you aren't fitting all the parameters well, whereas b is just a single number. So almost all the parameters are in w rather than b. And if you add this last term, in practice, it won't make much of a difference, because b is just one parameter over a very large number of parameters. 

![alt text](_assets/Regularization.png)

L2 regularization is the most common type of regularization. You might have also heard of some people talk about L1 regularization. And that's when you add, instead of this L2 norm, you instead add a term that is lambda over m of sum over, of this. 

${\lambda \over 2m}\Sigma_{j=1}^{n_x} |w| = {\lambda \over 2m} ||w||_1$

If you use L1 regularization, then w will end up being sparse. And what that means is that the w vector will have a lot of zeros in it. And some people say that this can help with compressing the model, because the set of parameters are zero, then you need less memory to store the model. Although, I find that, in practice, L1 regularization, to make your model sparse, helps only a little bit. So I don't think it's used that much, at least not for the purpose of compressing your model. And when people train your networks, L2 regularization is just used much, much more often. 

$\lambda$ is regularization parameter. And usually, you set this using your development set, or using hold-out cross validation. When you try a variety of values and see what does the best, in terms of trading off between doing well in your training set versus also setting that two normal of your parameters to be small, which helps prevent over fitting. So lambda is another hyper parameter that you might have to tune. And by the way, for the programming exercises, lambda is a reserved keyword in the Python programming language. So we use lambd so as not to clash with the reserved keyword in Python.

![alt text](_assets/L2LogisticRegression.png)

In Neural Network

$J(w^{[1]}, b^{[1]}, ..., w^{[L]}, b^{[L]}) = {1 \over m}\Sigma_{i=1}^m \ell(\hat{y}^{(i)}, y^{(i)}) + {\lambda \over 2m} \Sigma_{l=1}^L||w^{[l]}||^2$

$||w^{[l]}||^2_F = \Sigma_{i=1}^{n^{[l]}} \Sigma_{j=1}^{n^{[l-1]}}(w^{[l]}_{ij})^2$

w : $(n^{[l]}, n^{[l-1]})$ dimensional matrix

The rows "i" of the matrix should be the number of neurons in the current layer $n^{[l]}$.

The columns "j" of the weight matrix should equal the number of neurons in the previous layer $n^{[l-1]}$.

So this matrix norm, it turns out is called the "Frobenius norm" of the matrix, denoted with a F in the subscript.

For arcane linear algebra technical reasons, this is not called the, you know, l2 norm of a matrix. Instead, it's called the Frobenius norm of a matrix. I know it sounds like it would be more natural to just call the l2 norm of the matrix, but for really arcane reasons that you don't need to know, by convention, this is called the Frobenius norm. It just means the sum of square of elements of a matrix.

Compute gradient descent
* $dw^{[l]} = (from backprop) + {\lambda \over m} w^{[l]}$
* $w^{[l]} := w^{[l]} - \alpha ((from backprop) + {\lambda \over m} w^{[l]})$
  * $w^{[l]} := w^{[l]} - {\alpha \lambda \over m}w^{[l]} - \alpha (frombackprop)$

backprop is partial derivative of J with respect to w.

L2 regularization is sometimes called "weight decay".

This term shows that whatever the matrix $w^{[l]}$ is, you're going to make it a little bit smaller. This is actually as if you're taking the matrix w and you're multiplying it by 1 minus alpha lambda over m. ($(1-{\alpha \lambda \over m})$)

You're really taking the matrix w and subtracting alpha lambda over m times this. Like you're multiplying the matrix w by this number, which is going to be a little bit less than 1. 

So this is why L2 norm regularization is also called weight decay. Because it's just like the ordinary gradient descent, where you update w by subtracting alpha, times the original gradient you got from backprop. But now you're also, you know, multiplying w by this thing, which is a little bit less than 1. So the alternative name for L2 regularization is weight decay. I'm not really going to use that name, but the intuition for why it's called weight decay is that $(1-{\alpha \lambda \over m})w^{[l]} = w^{[l]} - {\alpha \lambda \over m}w^{[l]}$. So you're just multiplying the weight matrix by a number slightly less than 1. 

![alt text](_assets/L2NN.png)

## Why Regularization Reduces Overfitting?
One piece of intuition is that if you crank your regularization lambda to be really, really big, that'll be really incentivized to set the weight matrices, W, to be reasonably close to zero. So one piece of intuition is maybe it'll set the weight to be so close to zero for a lot of hidden units that's basically zeroing out a lot of the impact of these hidden units. And if that's the case, then, you know, this much simplified neural network becomes a much smaller neural network. In fact, it is almost like a logistic regression unit, you know, but stacked multiple layers deep. 

![alt text](_assets/SimpleNN.png)

And so that will take you from this overfitting case, much closer to the left, to the other high bias case. But, hopefully, there'll be an intermediate value of lambda that results in the result closer to this "just right" case in the middle.

![alt text](_assets/RegularizationExplained.png)

The intuition is that by cranking up lambda to be really big, it'll set W close to zero, which, in practice, this isn't actually what happens. We can think of it as zeroing out, or at least reducing, the impact of a lot of the hidden units, so you end up with what might feel like a simpler network, that gets closer and closer as if you're just using logistic regression. 

The intuition of completely zeroing out a bunch of hidden units isn't quite right. It turns out that what actually happens is it'll still use all the hidden units, but each of them would just have a much smaller effect. But you do end up with a simpler network, and as if you have a smaller network that is, therefore, less prone to overfitting. 

![alt text](_assets/tanhActivation.png)

g(z) = tanh(Z)

Notice that so long as z is quite small, so if z takes on only a smallish range of parameters, maybe around here

![alt text](_assets/tanh_SmallZ.png)

Then you're just using the linear regime of the tanh function, is only if z is allowed to wander, you know, to larger values or smaller values like so, that the activation function starts to become less linear. So the intuition you might take away from this is that if lambda, the regularization parameter is large, then you have that your parameters will be relatively small, because they are penalized being large in the cost function. And so if the weights, W, are small, then because z is equal to W, right, and then technically, it's plus b. 

$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$

But if W tends to be very small, then z will also be relatively small. And in particular, if z ends up taking relatively small values, just in this little range, then g of z will be roughly linear. So it's as if every layer will be roughly linear, as if it is just linear regression. And we saw in course one that if every layer is linear, then your whole network is just a linear network. And so even a very deep network, with a deep network with a linear activation function is, at the end of the day, only able to compute a linear function. So it's not able to, you know, fit those very, very complicated decision, very non-linear decision boundaries that allow it to really overfit to data sets, like we saw on the overfitting high variance case on the previous slide.

So just to summarize, if the regularization parameters are very large, the parameters W very small, so z will be relatively small, kind of ignoring the effects of b for now, but so z is relatively, so z will be relatively small, or really, I should say it takes on a small range of values. And so the activation function if it's tan h, say, will be relatively linear. And so your whole neural network will be computing something not too far from a big linear function, which is therefore, pretty simple function, rather than a very complex highly non-linear function. And so, is also much less able to overfit.

![alt text](_assets/RegularizationPreventOverfit.png)

When implementing regularization, we took our definition of the cost function J and we actually modified it by adding this extra term that penalizes the weights being too large. And so if you implement gradient descent, one of the steps to debug gradient descent is to plot the cost function J, as a function of the number of elevations of gradient descent, and you want to see that the cost function J decreases monotonically after every elevation of gradient descent. And if you're implementing regularization, then please remember that J now has this new definition. 

![alt text](_assets/Regularization_Gradient.png)

If you plot the old definition of J, just this first term, then you might not see a decrease monotonically. So to debug gradient descent, make sure that you're plotting this new definition of J that includes this second term as well.

## Dropout Regularization
![alt text](_assets/NNExample.png)

Let's say we train a NN and there is over-fitting.

With dropout, what we're going to do is go through each of the layers of the network and set some probability of eliminating a node in neural network. Let's say that for each of these layers, we're going to- for each node, toss a coin and have a 0.5 chance of keeping each node and 0.5 chance of removing each node. So, after the coin tosses, maybe we'll decide to eliminate those nodes, then what you do is actually remove all the outgoing things from that no as well.

![alt text](_assets/NNExampleDropout.png)

So you end up with a much smaller, really much diminished network. And then you do back propagation training.

![alt text](_assets/NNExampleDropout2.png)

And then on different examples, you would toss a set of coins again and keep a different set of nodes and then dropout or eliminate different set of nodes. And so for each training example, you would train it using one of these new networks.

Maybe it seems like a slightly crazy technique. They just go around coding those are random, but this actually works. But you can imagine that because you're training a much smaller network on each example, maybe just give a sense for why you end up able to regularize the network, because these much smaller networks are being trained.

### Implement dropout ("Inverted dropout")
Illustrate with layer l = 3

Let d3 to be the dropout vector for the layer 3

d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep_prop

keep_prob is a number. It was 0.5 on the previous time, and maybe now I'll use 0.8 in this example, and it will be the probability that a given hidden unit will be kept. keep.prob = 0.8, then this means that there's a 0.2 chance of eliminating any hidden unit.

We generate the same-shaped matrix of uniform random numbers in [0,1), for example:
```lua
rand =
[[0.549, 0.715, 0.603],
 [0.545, 0.424, 0.646],
 [0.438, 0.892, 0.964],
 [0.383, 0.792, 0.529],
 [0.568, 0.926, 0.071]]
```
Then compare each element with keep_prob = 0.8 using <:
```python
d3_bool = rand < keep_prob
```
This yields a boolean mask: True when rand < 0.8 (keep), False when rand ≥ 0.8 (drop).

```mathematica
d3_bool =
[[ True  True  True],
 [ True  True  True],
 [ True False False],
 [ True False  True],
 [ True False  True]]
```
As integers (1=keep, 0=drop):

```lua
d3_int =
[[1 1 1],
 [1 1 1],
 [1 0 0],
 [1 0 1],
 [1 0 1]]
```

Each random number is independent and uniform in [0,1). P(rand < keep_prob) = keep_prob. So each element independently has probability keep_prob to be True (kept).

What it does is it generates a random matrix. And this works as well if you have vectorized. So d3 will be a matrix. Therefore, each example have a each hidden unit there's a 0.8 chance that the corresponding d3 will be one, and a 20% chance there will be zero. So, this random numbers being less than 0.8 it has a 0.8 chance of being 1 or be true, and 20% or 0.2 chance of being false, of being 0. 

And then what you are going to do is take your activations from the third layer, let me just call it a3 in this example. 

a3 = np.multiply(a3,d3) # a3 *= d3

What this does is for every element of d3 that's equal to zero. And there was a 20% chance of each of the elements being zero, just multiply operation ends up zeroing out the corresponding element of d3.

```ini
a3 =
[[0.2 0.4 0.1]
 [0.5 0.6 0.2]
 [0.9 0.1 0.3]
 [0.0 0.7 0.8]
 [0.3 0.2 0.9]]
```
This zeros out dropped neurons:
```lua
a3_dropped = a3 * d3_int =
[[0.2 0.4 0.1],
 [0.5 0.6 0.2],
 [0.9 0.0 0.0],
 [0.0 0.0 0.8],
 [0.3 0.0 0.9]]
```

Notice entries where d3_int is 0 became 0 in a3_dropped.

If you do this in python, technically d3 will be a boolean array where value is true and false, rather than 1 and 0. But the multiply operation works and will interpret the true and false values as 1 and 0. 

a3 /= keep_prop

We divide the remaining (kept) activations by keep_prob:
```lua
a3_scaled = a3_dropped / 0.8 =
[[0.25  0.5   0.125],
 [0.625 0.75  0.25 ],
 [1.125 0.    0.   ],
 [0.    0.    1.00 ],
 [0.375 0.    1.125]]
```

Let say we have 50 units or 50 neurons in the 3rd hidden layer. So a3 is 50 by 1 dimensional or 50 by m dimensional if we vectorize it. If we have a 80% chance of keeping them and 20% chance of eliminating them, this means on average, you end up with 10 units shut off and 10 units zeroed out.

$z^{[4]} = W^{[4]}a^{[3]} + b^{[4]}$

On expectation, $a^{[3]}$ will be reduced by 20%, meaning 20% of a3 will be zeroed out.

In order to not reduce the expected value of $z^{[4]}$, what you do is you need to take $a^{[3]}$, and divide it by 0.8 because this will correct or just a bump that back up by roughly 20% that you need. So it does not change the expected value of a3. 

And, so this line here is what's called the inverted dropout technique. And its effect is that, no matter what you set to keep_prob to, whether it's 0.8 or 0.9 or even 1, if it's set to 1 then there's no dropout, because it's keeping everything 0.5 or whatever, this inverted dropout technique by dividing by the keep_prob, it ensures that the expected value of a3 remains the same.

It turns out that at test time, when you trying to evaluate a neural network, this inverted dropout technique makes test time easier because you have less of a scaling problem. 

By far the most common implementation of dropouts today as far as I know is inverted dropouts. 

But there were some early iterations of dropout that missed this divide by keep_prob line, and so at test time the average becomes more and more complicated. But again, people tend not to use those other versions. 

So, what you do is you use the d vector, and you'll notice that for different training examples, you zero out different hidden units. And in fact, if you make multiple passes through the same training set, then on different passes through the training set, you should randomly zero out different hidden units. So, it's not that for one example, you should keep zeroing out the same hidden units is that, on iteration one of grade and descent, you might zero out some hidden units. And on the second iteration of great descent where you go through the training set the second time, maybe you'll zero out a different pattern of hidden units. And the vector d or d3, for the third layer, is used to decide what to zero out, both in forward prop as well as in back prop. We are just showing forward prop here.

### Making predictions at test time

At test time, you're given some X of which you want to make a prediction.

$a^{[0]} = X$

Not use dropout at test time

* $z^{[1]} = W^{[1]}a^{[0]} + b^{[1]}$
* $a^{[1]} = g^{[1]}(z^{[1]})$
* $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$
* $a^{[2]} = g^{[2]}(z^{[2]})$
* So on until we get to the last layer and that we make a prediction with $\hat{y}$

But notice that the test time you're not using dropout explicitly and you're not tossing coins at random, you're not flipping coins to decide which hidden units to eliminate. And that's because when you are making predictions at the test time, you don't really want your output to be random. 

If you are implementing dropout at test time, that just add noise to your predictions. In theory, one thing you could do is run a prediction process many times with different hidden units randomly dropped out and have it across them. But that's computationally inefficient and will give you roughly the same result; very, very similar results to this different procedure as well. And just to mention, the inverted dropout thing, you remember the step on the previous line when we divided by the keep_prob. The effect of that was to ensure that even when you don't see men dropout at test time to the scaling, the expected value of these activations don't change. So, you don't need to add in an extra funny scaling parameter at test time. That's different than when you have that training time.

Dropout is a regularization technique used to prevent overfitting.

In training, we randomly turn off some neurons (set their activations to 0) so the network doesn’t depend too much on any single neuron. During testing (inference), we use all neurons — but scaled appropriately.

The “inverted” part simply means: We scale the remaining neurons during training, so we don’t have to do any scaling at test time.

#### Why divide by keep_prob? (the expectation argument)

Let a be a single activation value, and let d be the dropout mask for that element (1 with probability p=keep_prob, 0 otherwise). After applying inverted dropout we get:

![alt text](_assets/invertedDropout.png)

Take expectation over the randomness of d:

![alt text](_assets/Expectation.png)

So the expected (average) post-dropout activation equals the original activation a. This ensures that, on average, the scale of activations is the same during training (so we don't have to rescale at test time).

## Understanding Dropout
Drop out randomly knocks out units in your network. So it's as if on every iteration you're working with a smaller neural network. And so using a smaller neural network seems like it should have a regularizing effect. 

Let's look at it from the perspective of a single unit. N

![alt text](_assets/SingleUnit.png)

For this unit to do his job has four inputs and it needs to generate some meaningful output. Now with drop out, the inputs can get randomly eliminated. Sometimes those two units will get eliminated. Sometimes a different unit will get eliminated. So what this means is that this unit which I'm circling purple.

It can't rely on anyone feature because anyone feature could go away at random or anyone of its own inputs could go away at random.

So in particular, I will be reluctant to put all of its bets on say just this input, right. The ways were reluctant to put too much weight on anyone input because it could go away. So this unit will be more motivated to spread out this ways and give you a little bit of weight to each of the four inputs to this unit. And by spreading out the weights this will tend to have an effect of shrinking the squared norm of the weights, and so similar to what we saw with L2 regularization. The effect of implementing dropout is that its strength the ways and similar to L2 regularization, it helps to prevent overfitting, but it turns out that dropout can formally be shown to be an adaptive form of L2 regularization, but the L2 penalty on different ways are different depending on the size of the activation is being multiplied into that way. But to summarize it is possible to show that dropout has a similar effect to. 

Deep nets with many hidden units can overfit: they learn spurious patterns tied to the training set that don’t generalize.

Some hidden units may co-adapt: they become overly reliant on each other.

Dropout breaks these co-adaptations by randomly “dropping” units during training, so each unit must work somewhat independently.

“Dropout makes a network behave like many different “thinned” networks and then average them at test time.”

Let’s imagine one neuron (let’s call it N) in your neural network.
This neuron has four inputs coming in — like this:

```css
input1 → \
input2 →  \
input3 →   → [ N ] → output
input4 →  /
```

Each input has a weight (W₁, W₂, W₃, W₄).
These weights decide how much the neuron listens to each input.

So normally, N computes:

```ini
output = W₁*x₁ + W₂*x₂ + W₃*x₃ + W₄*x₄
```

Now, dropout randomly turns off some inputs each time.

So sometimes:

```java
input1 and input3 are gone (dropped)
```

Other times:

```sql
input2 and input4 are gone
```

So for each training step, the neuron receives different combinations of inputs.

If you’re this neuron (N), you quickly learn:

“Hey, I can’t depend too much on any single input… because it might disappear next time!”

So instead of giving one input a huge weight (like W₃ = 10, and others = 0.1), you’ll spread your attention more evenly across all inputs.

That is, you’ll make:
```
W₁, W₂, W₃, W₄ more balanced
```

This spreading out of weights helps the network be less fragile and more general — it doesn’t collapse if one feature (input) is missing or noisy.

Let’s recall what L2 regularization (a.k.a. “weight decay”) does:

* It penalizes very large weights.
* This keeps the model from “overfitting” — memorizing data too precisely.

Now look at what dropout just did:
* It made the neuron reluctant to make any weight too large,
because that input could vanish at any time.
* So dropout ends up having the same effect: it reduces the size of weights, helping prevent overfitting.

That’s why Andrew says:

“Dropout is like an adaptive form of L2 regularization.”

The word adaptive means:

* The effect depends on how active each input is.
* Inputs that are often dropped or have smaller activations get slightly different penalty strengths — but the overall effect is like L2 regularization.

One more detail for when you're implementing dropout, here's a network where you have three input features. 

![alt text](_assets/NNDropoutExample.png)

This is seven hidden units here. 7, 3, 2, 1, so one of the practice we have to choose was the keep prop which is a chance of keeping a unit in each layer. So it is also feasible to vary keep-propped by layer. So for the first layer, your matrix W1 will be 7 by 3. Your second weight matrix will be 7 by 7. W3 will be 3 by 7 and so on. And so W2 is actually the biggest weight matrix, right? Because they're actually the largest set of parameters. B and W2, which is 7 by 7. So to prevent, to reduce overfitting of that matrix, maybe for this layer, I guess this is layer 2, you might have a key prop that's relatively low, say 0.5, whereas for different layers where you might worry less about over 15, you could have a higher key problem. Maybe just 0.7, maybe this is 0.7. And then for layers we don't worry about overfitting at all. You can have a key prop of 1.0. Right? So, you know, for clarity, these are numbers I'm drawing in the purple boxes. These could be different key props for different layers. Notice that the key problem 1.0 means that you're keeping every unit, and so you're really not using drop out for that layer. But for layers where you're more worried about overfitting really the layers with a lot of parameters you could say keep_prop to be smaller to apply a more powerful form of dropout. It's kind of like cranking up the regularization. Parameter lambda of L2 regularization where you try to regularize some layers more than others.

Technically you can also apply drop out to the input layer where you can have some chance of just acting out one or more of the input features, although in practice, usually don't do that often. And so key problem of 1.0 is quite common for the input there. You might also use a very high value, maybe 0.9 but is much less likely that you want to eliminate half of the input features. So usually keep_prop, if you apply that all will be a number close to 1. If you even apply dropout at all to the input layer. 

To summarize, if you're more worried about some layers of fitting than others, you can set a lower key prop for some layers than others. The downside is this gives you even more hyperparameters to search for using cross validation. One other alternative might be to have some layers where you apply dropout and some layers where you don't apply drop out and then just have one hyper parameter which is a keep_prop for the layers for which you do apply drop out.

Many of the first successful implementations of dropouts were to computer vision, so in computer vision, the input sizes so big in putting all these pixels that you almost never have enough data. And so drop out is very frequently used by the computer vision and there are some common vision research is that pretty much always use it almost as a default. But really, the thing to remember is that drop out is a regularization technique, it helps prevent overfitting. And so unless my algorithm is overfitting, I wouldn't actually bother to use drop out. 

So as you somewhat less often in other application areas, there's just a computer vision, you usually just don't have enough data so you almost always overfitting, which is why they tend to be some computer vision researchers swear by drop out by the intuition.

One big downside of drop out is that the cost function J is no longer well defined on every iteration. You're randomly, calling off a bunch of notes. And so if you are double checking the performance of gradient descent is actually harder to double check that you have a well defined cost function J that is going downhill on every iteration. Because the cost function J that you're going to optimizing is actually less well defined and it's certainly hard to calculate. So you lose this debugging tool to have a plot a draft like this. So what I usually do is turn off drop out or if you will set keep-prop = 1 and run my code and make sure that it is monitored quickly decreasing J. And then turn on drop out and hope that, I didn't introduce bug to my code during drop out because you need other ways, I guess, but not plotting these figures to make sure that your code is working, the gradient descent is working even with drop out. 

![alt text](_assets/CostJ.png)

**Increasing keep_prob (e.g. from 0.5 → 0.6):**
* Keeps more neurons active
* Makes dropout weaker (less regularization)
* May cause higher variance (model could overfit more)
* Slightly reduces bias (model fits training set better)

|Keep_prob|	Regularization strength|	Effect|
|-|-|-|
|Low (e.g. 0.3–0.5)|	Strong regularization|	Helps prevent overfitting (reduces variance), but may underfit (higher bias)|
|High (e.g. 0.6–0.9)|	Weak regularization|	Less dropout noise → model fits training data better (lower bias), but might overfit (higher variance)|

* Smaller keep_prob = stronger dropout (more neurons turned off)
* Larger keep_prob = weaker dropout (fewer neurons turned off)

## Other Regularization Methods

Let's say you fitting a cat classifier. If you are over fitting getting more training data can help, but getting more training data can be expensive and sometimes you just can't get more data. 

But what you can do is augment your training set by taking image like this. And for example, flipping it horizontally and adding that also with your training set. So now instead of just this one example in your training set, you can add this to your training example. 

![alt text](_assets/CatDataAugmentation.png)

So by flipping the images horizontally, you could double the size of your training set. Because you're training set is now a bit redundant this isn't as good as if you had collected an additional set of brand new independent examples. But you could do this Without needing to pay the expense of going out to take more pictures of cats. 

And then other than flipping horizontally, you can also take random crops of the image. So here we're rotated and sort of randomly zoom into the image and this still looks like a cat. So by taking random distortions and translations of the image you could augment your data set and make additional fake training examples. Again, these extra fake training examples they don't add as much information as they were to get a brand new independent example of a cat. But because you can do this, almost for free, other than for some computational costs. This can be an inexpensive way to give your algorithm more data and therefore sort of regularize it and reduce over fitting. 

Notice I didn't flip it vertically, because maybe we don't want upside down cats. And then also maybe randomly zooming in to part of the image it's probably still a cat. 

For optical character recognition you can also bring your data set by taking digits and imposing random rotations and distortions to it. So If you add these things to your training set, these are also still digit force.

![alt text](_assets/DigitDataAugmentation.png)

For illustration I applied a very strong distortion. So this look very wavy number four, in practice you don't need to distort the four quite as aggressively, but just a more subtle distortion than what I'm showing here, to make this example clearer for you. But a more subtle distortion is usually used in practice, because this looks like really warped fours. So data augmentation can be used as a regularization technique, in fact similar to regularization. 

There's one other technique that is often used called early stopping. So what you're going to do is as you run gradient descent you're going to plot your, either the training error, you'll use 0 - 1 classification error on the training set. Or just plot the cost function J optimizing, and that should decrease monotonically.

![alt text](_assets/trainingErr.png)

So with early stopping, what you do is you plot this, and you also plot your dev set error. And again, this could be a classification error in a development sense, or something like the cost function, like the logistic loss or the log loss of the dev set. 

![alt text](_assets/devSetErr.png)

Now what you find is that your dev set error will usually go down for a while, and then it will increase from there. So what early stopping does is, you will say well, it looks like your neural network was doing best around that iteration, so we just want to stop training on your neural network halfway and take whatever value achieved this dev set error.

So why does this work? Well when you've haven't run many iterations for your neural network yet your parameters w will be close to zero. 

![alt text](_assets/wCloseTo0.png)

Because with random initialization you probably initialize w to small random values so before you train for a long time, w is still quite small.

And as you iterate, as you train, w will get bigger and bigger and bigger until here maybe you have a much larger value of the parameters w for your neural network. 

![alt text](_assets/largeW.png)

So what early stopping does is by stopping halfway you have only a mid-size rate w. 

![alt text](_assets/midSizeW.png)

And so similar to L2 regularization by picking a neural network with smaller norm for your parameters w, hopefully your neural network is over fitting less. And the term early stopping refers to the fact that you're just stopping the training of your neural network earlier. 

I sometimes use early stopping when training a neural network. But it does have 1 downside.

ML process comprises several steps:
1. An algorithm to optimize the cost function J and we have various tools to do that, such as gradien descent. (There are also momentum, RMS prop, Atom and so on)
2. We want it to not overfit. We have some tools such as regularization, getting more data, so on...

Now in machine learning, we already have so many hyper-parameters it surge over. It's already very complicated to choose among the space of possible algorithms. 

And so I find machine learning easier to think about when you have one set of tools for optimizing the cost function J, and when you're focusing on authorizing the cost function J. All you care about is finding w and b, so that J(w,b) is as small as possible. You just don't think about anything else other than reducing this. 

And then it's completely separate task to not over fit, in other words, to reduce variance. And when you're doing that, you have a separate set of tools for doing it. 

This principle is sometimes called **orthogonalization**. And there's this idea, that you want to be able to think about one task at a time. 

The main downside of early stopping is that this couples these two tasks. So you no longer can work on these two problems independently, because by stopping gradient decent early, you're sort of breaking whatever you're doing to optimize cost function J, because now you're not doing a great job reducing the cost function J. You've sort of not done that that well. And then you also simultaneously trying to not over fit. So instead of using different tools to solve the two problems, you're using one that kind of mixes the two. And this just makes the set of things you could try are more complicated to think about. 

![alt text](_assets/earlyStopping.png)

Rather than using early stopping, one alternative is just use L2 regularization then you can just train the neural network as long as possible. I find that this makes the search space of hyper parameters easier to decompose, and easier to search over. 

The downside of this though is that you might have to try a lot of values of the regularization parameter lambda. And so this makes searching over many values of lambda more computationally expensive. 

The advantage of early stopping is that running the gradient descent process just once, you get to try out values of small w, mid-size w, and large w, without needing to try a lot of values of the L2 regularization hyperparameter lambda.

Despite it's disadvantages, many people do use it. I personally prefer to just use L2 regularization and try different values of lambda. That's assuming you can afford the computation to do so. But early stopping does let you get a similar effect without needing to explicitly try lots of different values of lambda.

## Normalizing Inputs
When training a neural network, one of the techniques to speed up your training is if you normalize your inputs.

![alt text](_assets/2inputSet.png)

Let's see the training sets with two input features. The input features x are two-dimensional and here's a scatter plot of your training set.

Normalizing inputs corresponds to 2 steps.
1. Subtract out or zero out the mean, so sets mu equals 1 over m, sum over i of $x_i$. This is a vector and then x gets set as x minus mu for every training example.

![alt text](_assets/Step1.png)

This means that you just move the training set until it has zero mean. 

![alt text](_assets/zeroMean.png)

2. Normalize the variances. Notice here that the feature x_1 has a much larger variance than the feature x_2 here. What we do is set sigma equals 1 over m sum of x_i star, star 2. I guess this is element-wise squaring. Now sigma squared is a vector with the variances of each of the features. Notice we've already subtracted out the mean, so x_i squared, element-wise square is just the variances. You take each example and divide it by this vector sigma.

![alt text](_assets/Step2.png)

![alt text](_assets/Step2Picture.png)

Now the variance of x_1 and x_2 are both equal to one.

One tip. If you use this to scale your training data, then use the same mu and sigma to normalize your test set. In particular, you don't want to normalize the training set and a test set differently. Whatever this value is and whatever this value is, use them in these two formulas so that you scale your test set in exactly the same way rather than estimating mu and sigma squared separately on your training set and test set, because you want your data both training and test examples to go through the same transformation defined by the same Mu and Sigma squared calculated on your training data.

![alt text](_assets/normalizeTrainingSet.png)

Why do we do this? Why do we want to normalize the input features?

![alt text](_assets/costFunction.png)

It turns out that if you use unnormalized input features, it's more likely that your cost function will look like this, like a very squished out bar, very elongated cost function where the minimum you're trying to find is maybe over there.

![alt text](_assets/unnormalized.png)

But if your features are on very different scales, say the feature x_1 ranges from 1-1,000 and the feature x_2 ranges from 0-1, then it turns out that the ratio or the range of values for the parameters w_1 and w_2 will end up taking on very different values. Maybe these axes should be w_1 and w_2, but the intuition of plot w and b, cost function can be very elongated bow like that. If you plot the contours of this function, you can have a very elongated function like that. 

![alt text](_assets/unnormalizedContour.png)

Whereas if you normalize the features, then your cost function will on average look more symmetric. 

![alt text](_assets/normalized.png)

If you are running gradient descent on a cost function like the unnormalized one, then you might have to use a very small learning rate because if you're here, the gradient decent might need a lot of steps to oscillate back and forth before it finally finds its way to the minimum. 

![alt text](_assets/unnormalizedContour.png)

Whereas if you have more spherical contours, then wherever you start, gradient descent can pretty much go straight to the minimum. You can take much larger steps where gradient descent need, rather than needing to oscillate around like the picture on the left. Of course, in practice, w is a high dimensional vector.

![alt text](_assets/normalizedContour.png)

Trying to plot this in 2D doesn't convey all the intuitions correctly. But the rough intuition that you cost function will be in a more round and easier to optimize when you're features are on similar scales. Not from 1-1000, 0-1, but mostly from -1 to 1 or about similar variance as each other. That just makes your cost function j easier and faster to optimize.

In practice, if one feature, say x_1 ranges from 0-1 and x_2 ranges from -1 to 1, and x_3 ranges from 1-2, these are fairly similar ranges, so this will work just fine, is when they are on dramatically different ranges like ones from 1-1000 and another from 0-1. That really hurts your optimization algorithm. That by just setting all of them to zero mean and say variance one like we did on the last slide, that just guarantees that all your features are in a similar scale and will usually help you learning algorithm run faster.

If your input features came from very different scales, maybe some features are from 0-1, sum from 1-1000, then it's important to normalize your features. If your features came in on similar scales, then this step is less important although performing this type of normalization pretty much never does any harm. Often you'll do it anyway, if I'm not sure whether or not it will help with speeding up training for your algorithm. That's it for normalizing your input features.

## Vanishing / Exploding Gradients
One of the problems of training neural network, especially very deep neural networks, is data vanishing and exploding gradients. What that means is that when you're training a very deep network your derivatives or your slopes can sometimes get either very, very big or very, very small, maybe even exponentially small, and this makes training difficult. 

![alt text](_assets/deepNNExample.png)

This neural network will have parameters W1, W2, W3 and so on up to WL. For the sake of simplicity, let's say we're using an activation function G of Z equals Z, so linear activation function. And let's ignore B, let's say B of L equals zero. 

![alt text](_assets/deepNNExampleInfo.png)

In that case you can show that the output Y will be WL times WL minus one times WL minus two, dot, dot, dot down to the W3, W2, W1 times X. 

If you want to just check my math, W1 times X is going to be Z1, because B is equal to zero. So Z1 is equal to, I guess, W1 times X and then plus B which is zero. But then A1 is equal to G of Z1. But because we use linear activation function, this is just equal to Z1. So this first term W1X is equal to A1. And then by the reasoning you can figure out that W2 times W1 times X is equal to A2, because that's going to be G of Z2, is going to be G of W2 times A1 which you can plug that in here. 

![alt text](_assets/deepNNExampleFunc.png)

So this thing is going to be equal to A2, and then this thing is going to be A3 and so on until the protocol of all these matrices gives you Y-hat, not Y.

![alt text](_assets/deepNNExampleFunc2.png)

Let's say that each of you weight matrices WL is just a little bit larger than one times the identity. So it's 1.5_1.5_0_0. Technically, the last one has different dimensions so maybe this is just the rest of these weight matrices. 

![alt text](_assets/deepNNExampleW.png)

Then Y-hat will be, ignoring this last one with different dimension, this 1.5_0_0_1.5 matrix to the power of L minus 1 times X, because we assume that each one of these matrices is equal to this thing. It's really 1.5 times the identity matrix, then you end up with this calculation. And so Y-hat will be essentially 1.5 to the power of L, to the power of L minus 1 times X, and if L was large for very deep neural network, Y-hat will be very large. In fact, it just grows exponentially, it grows like 1.5 to the number of layers. And so if you have a very deep neural network, the value of Y will explode.

![alt text](_assets/deepNNExampleYhat.png)

Now, conversely, if we replace 1.5 with 0.5, so something less than 1, then this becomes 0.5 to the power of L. This matrix becomes 0.5 to the L minus 1 times X, again ignoring WL. 

![alt text](_assets/deepNNExampleReplaceValue.png)

And so each of your matrices are less than 1, then let's say X1, X2 were one one, then the activations will be one half, one half, one fourth, one fourth, one eighth, one eighth, and so on until this becomes one over two to the L. 

![alt text](_assets/deepNNExample2.png)

So the activation values will decrease exponentially as a function of the def, as a function of the number of layers L of the network. So in the very deep network, the activations end up decreasing exponentially. 

Intuition
1. If weight $W^{[l]}$ is bigger than 1 or bigger than identity matrix, then with a very deep network, the activations can explode. And when gradients (errors) flow backward through these same weights, they also get multiplied by those large numbers repeatedly, so gradients explode too — making updates unstable or overflow.
2. If weight $W^{[l]}$ is a little bit less than 1 or bigger than identity matrix, then with a very deep network, the activations can decrease exponentially. Gradients become tiny, Weight updates become so small that the network stops learning.

Even though I went through this argument in terms of activations increasing or decreasing exponentially as a function of L, a similar argument can be used to show that the derivatives or the gradients the computer is going to send will also increase exponentially or decrease exponentially as a function of the number of layers.

With some of the modern neural networks, L equals 150. Microsoft recently got great results with 152 layer neural network. But with such a deep neural network, if your activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or really small. And this makes training difficult, especially if your gradients are exponentially smaller than L, then gradient descent will take tiny little steps. It will take a long time for gradient descent to learn anything. 

## Weight Initialization for Deep Networks
In the last video you saw how very deep neural networks can have the problems of vanishing and exploding gradients. It turns out that a partial solution to this, doesn't solve it entirely but helps a lot, is **better or more careful choice of the random initialization for your neural network**. 

![alt text](_assets/SingleNeuronExample.png)

Let's go through this with an example with just a single neuron, and then we'll talk about the deep net later. So with a single neuron, you might input four features, x1 through x4, and then you have some a=g(z) and then it outputs some y. And later on for a deeper net, you know these inputs will be some layer $a^{[l]}$, but for now let's just call this x for now. So z is going to be equal to $w_1x_1 + w_2x_2 +... + W_nX_n$. 

Let's set b = 0.

In order to make z not blow up and not become too small, you notice that the larger n is, the smaller you want $W_i$ to be. Because z is the sum of the $W_iX_i$, so if you're adding up a lot of these terms, you want each of these terms to be smaller. 

One reasonable thing to do would be to set the variance of W to be equal to 1 over n, where n is the number of input features that's going into a neuron. 

$Var(W) = 1 \over n$

In practice

$W^{[l]}$ = np.random.randn(shape) * np.sqrt($1/n^{[l-1]}$)

because l-1 the number of units that I'm feeding into each of the units in layer l.

It turns out that if you're using a ReLu activation function that, rather than 1 over n it turns out that, set in the variance of 2 over n works a little bit better. 

$Var(W) = 2 \over n$

$W^{[l]}$ = np.random.randn(shape) * np.sqrt($2/n^{[l-1]}$)

So you often see that in initialization, especially if you're using a ReLu activation function. So if gl(z) = ReLu(z).

If the input features of activations are roughly mean 0 and standard variance, variance 1 then this would cause z to also take on a similar scale. And this doesn't solve, but it definitely helps reduce the vanishing, exploding gradients problem, because it's trying to set each of the weight matrices w, you know, so that it's not too much bigger than 1 and not too much less than 1 so it doesn't explode or vanish too quickly.

The version we just described is assuming a ReLu activation function. A few other variants, if you are using a TanH activation function then there's a paper that shows that instead of using the constant 2, it's better use the constant 1 and so 1 over this instead of 2. 

![alt text](_assets/tanhActivationWInit.png)

This is called Xavier initialization. 

![alt text](_assets/YoshuaInit.png)

Another version we're taught by Yoshua Bengio and his colleagues, you might see in some papers, but is to use this formula, which you know has some other theoretical justification, but I would say if you're using a ReLu activation function, which is really the most common activation function, I would use this formula.

![alt text](_assets/ReLUInit.png)

In practice I think all of these formulas just give you a starting point. It gives you a default value to use for the variance of the initialization of your weight matrices. If you wish the variance here, this variance parameter could be another thing that you could tune with your hyperparameters. So you could have another parameter that multiplies into this formula and tune that multiplier as part of your hyperparameter surge. 

Sometimes tuning the hyperparameter has a modest size effect. It's not one of the first hyperparameters I would usually try to tune, but I've also seen some problems where tuning this helps a reasonable amount. But this is usually lower down for me in terms of how important it is relative to the other hyperparameters you can tune.

## Numerical Approximation of Gradients
When you implement back propagation you'll find that there's a test called gradient checking that can really help you make sure that your implementation of back prop is correct. Because sometimes you write all these equations and you're just not 100% sure if you've got all the details right in implementing back propagation.

![alt text](_assets/FunctionF.png)

Let's take the function f and replot it here and remember this is f of theta equals theta cubed, and let's again start off to some value of theta. Let's say theta equals 1. Now instead of just nudging theta to the right to get theta plus epsilon, we're going to nudge it to the right and nudge it to the left to get theta minus epsilon, as was theta plus epsilon.

![alt text](_assets/FunctionFwithEpsilon.png)

It turns out that rather than taking this little triangle and computing the height over the width, you can get a much better estimate of the gradient if you take this point, f of theta minus epsilon and this point, f of theta plus epsilon, and you instead compute the height over width of this bigger triangle. 

![alt text](_assets/heightOverWidth.png)

The height over width of this bigger green triangle gives you a much better approximation to the derivative at theta. 

![alt text](_assets/heightOverWidth2.png)

Taking just this lower triangle in the upper right is as if you have two triangles. This one on the upper right and this one on the lower left. 

![alt text](_assets/2triangles.png)

And you're kind of taking both of them into account by using this bigger green triangle. So rather than a one sided difference, you're taking a two sided difference.

![alt text](_assets/biggerTriangle.png)

This point here is F of theta plus epsilon. This point here is F of theta minus epsilon. So the height of this big green triangle is f of theta plus epsilon minus f of theta minus epsilon. And then the width, this is 1 epsilon, this is 2 epsilon. So the width of this green triangle is 2 epsilon. So the height of the width is going to be first the height, so that's F of theta plus epsilon minus F of theta minus epsilon divided by the width. So that was 2 epsilon which we write that down here. 

![alt text](_assets/Math.png)

![alt text](_assets/Math2.png)

Plug in the values, remember f of theta is theta cubed. So this is theta plus epsilon is 1.01. So I take a cube of that minus 0.99 theta cube of that divided by 2 times 0.01. You should get that this is 3.0001.

![alt text](_assets/Math3.png)

## Gradient Checking
Take $W^{[1]}, b^{[1]}, ..., W^{[L]}, b^{[L]}$ and reshape into a big vector $\theta$.

What you should do is take W which is a matrix, and reshape it into a vector. You gotta take all of these Ws and reshape them into vectors, and then concatenate all of these things, so that you have a giant vector theta. Giant vector pronounced as $\theta$. 

We say that the cos function J being a function of the Ws and Bs, You would now have the cost function J being just a function of theta.

![alt text](_assets/JofTheta.png)

Take $dW^{[1]}, db^{[1]}, ..., dW^{[L]}, db^{[L]}$ and reshape into a big vector $d\theta$.

Same as before, we shape dW[1] into the matrix, db[1] is already a vector. We shape dW[L], all of the dW's which are matrices. Remember, dW1 has the same dimension as W1. db1 has the same dimension as b1. So the same sort of reshaping and concatenation operation, you can then reshape all of these derivatives into a giant vector d theta. Which has the same dimension as theta. 

Is $d\theta$ is gradient (or slope) of cost function J?

First we remember that J Is now a function of the giant parameter, theta. So expands to j is a function of theta 1, theta 2, theta 3, and so on. Whatever's the dimension of this giant parameter vector theta. To implement grad check, what you're going to do is implements a loop so that for each i, so for each component of theta, let's compute D theta approx i to b. And let me take a two sided difference. I'll take J of theta. Theta 1, theta 2, up to theta i. And we're going to nudge theta i to add epsilon to this. So just increase theta i by epsilon, and keep everything else the same. And because we're taking a two sided difference, we're going to do the same on the other side with theta i, but now minus epsilon. And then all of the other elements of theta are left alone. And then we'll take this, and we'll divide it by 2 theta.

![alt text](_assets/GradCheck.png)

What we saw from the previous video is that this should be approximately equal to d theta i. Of which is supposed to be the partial derivative of J or of respect to, I guess theta i, if d theta i is the derivative of the cost function J. So what you going to do is you're going to compute to this for every value of i. And at the end, you now end up with two vectors. You end up with this d theta approx, and this is going to be the same dimension as d theta. And both of these are in turn the same dimension as theta. And what you want to do is check if these vectors are approximately equal to each other.

![alt text](_assets/GradCheck2.png)

I would compute the distance between these two vectors, d theta approx minus d theta, so just the o2 norm of this. Notice there's no square on top, so this is the sum of squares of elements of the differences, and then you take a square root, as you get the Euclidean distance. And then just to normalize by the lengths of these vectors, divide by d theta approx plus d theta. Just take the Euclidean lengths of these vectors. 

![alt text](_assets/GradCheck3.png)

And the row for the denominator is just in case any of these vectors are really small or really large, your the denominator turns this formula into a ratio. 

In practice, I use epsilon equals maybe 10 to the minus 7. And with this range of epsilon, if you find that this formula gives you a value like 10 to the minus 7 or smaller, then that's great. It means that your derivative approximation is very likely correct. This is just a very small value. If it's maybe on the range of 10 to the -5, I would take a careful look. Maybe this is okay. But I might double-check the components of this vector, and make sure that none of the components are too large. And if some of the components of this difference are very large, then maybe you have a bug somewhere. And if this formula on the left is on the other is -3, then I would wherever you have would be much more concerned that maybe there's a bug somewhere. But you should really be getting values much smaller then 10 minus 3. If any bigger than 10 to minus 3, then I would be quite concerned. I would be seriously worried that there might be a bug. And I would then, you should then look at the individual components of data to see if there's a specific value of i for which d theta across i is very different from d theta i. And use that to try to track down whether or not some of your derivative computations might be incorrect. 

![alt text](_assets/GradCheck4.png)

And after some amounts of debugging, it finally, it ends up being this kind of very small value, then you probably have a correct implementation. 

So when implementing a neural network, what often happens is I'll implement foreprop, implement backprop. And then I might find that this grad check has a relatively big value. And then I will suspect that there must be a bug, go in debug, debug, debug. And after debugging for a while, If I find that it passes grad check with a small value, then you can be much more confident that it's then correct.

You already wrote a neural network that does:
1. Forward propagation → computes the cost J($\theta$)
2. Backward propagation → computes the gradients ${dJ} \over {d\theta}$

Here 𝜃 (“theta”) is the vector of all your parameters (weights & biases).

Now, we want to check if the backprop gradients are correct.

Backprop computes derivatives using the chain rule — fast but complex.

But we can also approximate derivatives using basic calculus:

![alt text](_assets/ApproxDerivative.png)

That’s called the numerical gradient.
* 𝜀 (epsilon) is a tiny number (like $10^{-7}$)
* You move a little bit forward and backward along one parameter, measure how the cost changes, and estimate the slope.

Imagine your cost function (loss) depends on only one weight w:

$J(w) = w^{2}$

You know the true gradient is: ${{dJ} \over {dw}}=2w$

Pick w = 3 and epsilon - 0.001.

$NumericalGradient \approx {{J(3+0.001) - J(3-0.001)} \over {2*0.001}}$

![alt text](_assets/ComputeStepByStep.png)

Theoretical gradient = 2*3 = 6.

In a real neural network, you have many parameters:

$\theta = [W_1,b_1,...,W_L,b_L]$

1. Flatten all parameters into a single vector theta.
2. Flatten all gradients (computed by backprop) into a single vector grad.
3. Then for each parameter i in theta:
  * Compute $J(\theta^+) = J(\theta + \epsilon e_i)$
  * Compute $J(\theta^-) = J(\theta - \epsilon e_i)$
  * Approximate derivative:
      ![alt text](_assets/ApproxDerivative2.png)
4. Compare dtheta_num (numerical gradients) with grad (backprop gradients).

You can’t expect exact equality (floating-point precision),
so you compute the difference ratio:

![alt text](_assets/difference.png)

If:
* difference < 1e-7 → ✅ Your backprop is correct
* difference ≈ 1e-5 to 1e-7 → ⚠️ Maybe okay
* difference > 1e-4 → ❌ You probably have a bug

![alt text](_assets/Example.png)

1. Don’t use dropout while checking gradients
  → Dropout adds randomness, so you’ll never get matching results.
1. Don’t use gradient checking every iteration
  → It’s slow. Just use it once after implementing backprop to confirm correctness.
1. Always reset ε small (like 1e−7)
  → If too large, approximation is bad; if too small, round-off errors dominate.
1. Turn off regularization or include it properly in J
  → If you regularize weights, include that in both the forward and numerical cost.

## Gradient Checking Implementation Notes

* Don’t use in training – only to debug

Computing $d \theta _{approx}[i]$, this is a very slow computation. So to implement gradient descent, you'd use backprop to compute d theta and just use backprop to compute the derivative. And it's only when you're debugging that you would compute this to make sure it's close to d theta. But once you've done that, then you would turn off the grad check, and don't run this during every iteration of gradient descent, because that's just much too slow. 

* If algorithm fails grad check, look at components to try to identify bug.

If $d \theta _{approx}[i]$ is very far from $d \theta$, what I would do is look at the different values of i to see which are the values of $d \theta _{approx}[i]$ that are really very different than the values of d theta. So for example, if you find that the values of theta or d theta, they're very far off, all correspond to $db^{[l]}$ for some layer or for some layers, but the components for $dw^{[l]}$ are quite close. Remember, different components of theta correspond to different components of b and w. When you find this is the case, then maybe you find that the bug is in how you're computing db, the derivative with respect to parameters b. And similarly, vice versa, if you find that the values that are very far, the values from $d \theta _{approx}[i]$ that are very far from d theta, you find all those components came from dw or from dw in a certain layer, then that might help you hone in on the location of the bug. This doesn't always let you identify the bug right away, but sometimes it helps you give you some guesses about where to track down the bug.

* Remember regularization.

If your cost function is J of theta equals 1 over m sum of your losses and then plus this regularization term. And sum over l of wl squared, then this is the definition of J. And you should have that d theta is gradient of J with respect to theta, including this regularization term. So just remember to include that term. 

![alt text](_assets/Regularization2.png)

* Grad check doesn’t work with dropout.

Because in every iteration, dropout is randomly eliminating different subsets of the hidden units. There isn't an easy to compute cost function J that dropout is doing gradient descent on. It turns out that dropout can be viewed as optimizing some cost function J, but it's cost function J defined by summing over all exponentially large subsets of nodes they could eliminate in any iteration. So the cost function J is very difficult to compute, and you're just sampling the cost function every time you eliminate different random subsets in those we use dropout. So it's difficult to use grad check to double check your computation with dropouts. So what I usually do is implement grad check without dropout. So if you want, you can set keep_prop and dropout to be equal to 1.0. And then turn on dropout and hope that my implementation of dropout was correct.

So my recommendation is turn off dropout, use grad check to double check that your algorithm is at least correct without dropout, and then turn on dropout.

* Run at random initialization; perhaps again after some training.

Finally, this is a subtlety. It is not impossible, rarely happens, but it's not impossible that your implementation of gradient descent is correct when w and b are close to 0, so at random initialization. But that as you run gradient descent and w and b become bigger, maybe your implementation of backprop is correct only when w and b is close to 0, but it gets more inaccurate when w and b become large. So one thing you could do, I don't do this very often, but one thing you could do is run grad check at random initialization and then train the network for a while so that w and b have some time to wander away from 0, from your small random initial values. And then run grad check again after you've trained for some number of iterations.


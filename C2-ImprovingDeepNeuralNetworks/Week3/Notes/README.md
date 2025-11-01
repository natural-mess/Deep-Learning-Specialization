# Hyperparameter Tuning, Batch Normalization and Programming Frameworks

**Learning Objectives**
* Master the process of hyperparameter tuning
* Describe softmax classification for multiple classes
* Apply batch normalization to make your neural network more robust
* Build a neural network in TensorFlow and train it on a TensorFlow dataset
* Describe the purpose and operation of GradientTape
* Use tf.Variable to modify the state of a variable
* Apply TensorFlow decorators to speed up code
* Explain the difference between a variable and a constant

## Tuning Process
### Hyperparameters
One of the painful things about training deepness is the sheer number of hyperparameters you have to deal with:
* Learning rate $\alpha$
* Momentum term $\beta$
* Hyperparameters for the Adam Optimization Algorithm which are $\beta_1$, $\beta_2$, and $\epsilon$.
* Number of layers
* Number of hidden units for the different layers
* Learning rate decay, so you don't just use a single learning rate $\alpha$. And then of course
* Mini-batch size. 

Some of these hyperparameters are more important than others. For most learning applications
1. $\alpha$, the learning rate is the most important hyperparameter to tune. 
2. Other than alpha, a few other hyperparameters I tend to would maybe tune next, would be maybe **the momentum term**, 0.9 is a good default. 
3. I'd also tune the **mini-batch size** to make sure that the optimization algorithm is running efficiently.
4. Often I also fiddle around with the **hidden units**.

Of the ones om 2, 3, 4, these are really the three that I would consider second in importance to the learning rate $\alpha$

5. Third in importance after fiddling around with the others, the **number of layers** can sometimes make a huge difference, and so can **learning rate decay**. 
6. And then, when using the Adam algorithm I actually pretty much never tuned $\beta_1$, $\beta_2$, and $\epsilon$. Pretty much I always use 0.9, 0.999 and $10^{-8}$ although you can try tuning those as well if you wish.

### Try random values: Don’t use a grid
If you're trying to tune some set of hyperparameters, how do you select a set of values to explore?

In earlier generations of machine learning algorithms, if you had two hyperparameters, which I'm calling hyperparameter 1 and hyperparameter 2 here, it was common practice to sample the points in a grid like so, and systematically explore these values. Here I am placing down a five by five grid. In practice, it could be more or less than the five by five grid but you try out in this example all 25 points, and then pick whichever hyperparameter works best.

![alt text](_assets/2HyperparametersGrid.png)

This practice works okay when the number of hyperparameters was relatively small.

In deep learning, what we tend to do, and what I recommend you do instead, is choose the points at random. So go ahead and choose maybe of same number of points, right? 25 points, and then try out the hyperparameters on this randomly chosen set of points.

![alt text](_assets/2HyperparametersGridRandom.png)

The reason you do that is that it's difficult to know in advance which hyperparameters are going to be the most important for your problem. As you saw in the previous slide, some hyperparameters are actually much more important than others.

So to take an example, let's say hyperparameter one turns out to be alpha, the learning rate. And to take an extreme example, let's say that hyperparameter two was that value epsilon that you have in the denominator of the Adam algorithm. So your choice of alpha matters a lot and your choice of epsilon hardly matters. So if you sample in the grid then you've really tried out five values of alpha and you might find that all of the different values of epsilon give you essentially the same answer. So you've now trained 25 models and only got into trial five values for the learning rate alpha, which I think is really important.

Whereas in contrast, if you were to sample at random, then you will have tried out 25 distinct values of the learning rate alpha and therefore you be more likely to find a value that works really well.

I've explained this example, using just two hyperparameters. In practice, you might be searching over many more hyperparameters than these, so if you have, say, three hyperparameters, I guess instead of searching over a square, you're searching over a cube where this third dimension is hyperparameter three and then by sampling within this three-dimensional cube you get to try out a lot more values of each of your three hyperparameters. 

![alt text](_assets/3HyperparametersRandom.png)

In practice you might be searching over even more hyperparameters than three and sometimes it's just hard to know in advance which ones turn out to be the really important hyperparameters for your application and sampling at random rather than in the grid shows that you are more richly exploring set of possible values for the most important hyperparameters, whatever they turn out to be.

### Coarse to fine
When you sample hyperparameters, another common practice is to use a coarse to fine sampling scheme.

Let's say in this two-dimensional example that you sample these points, and maybe you found that this point work the best and maybe a few other points around it tended to work really well, then in the course of the final scheme what you might do is zoom in to a smaller region of the hyperparameters, and then sample more density within this space. 

Or maybe again at random, but to then focus more resources on searching within this blue square if you're suspecting that the best setting, the hyperparameters, may be in this region. So after doing a coarse sample of this entire square, that tells you to then focus on a smaller square. You can then sample more densely into smaller square. So this type of a coarse to fine search is also frequently used. 

![alt text](_assets/coarseToFine.png)

And by trying out these different values of the hyperparameters you can then pick whatever value allows you to do best on your training set objective, or does best on your development set, or whatever you're trying to optimize in your hyperparameter search process.

The two key takeaways are:
* Use random sampling and adequate search.
* Optionally consider implementing a coarse to fine search process.

## Using an Appropriate Scale to pick Hyperparameters
### Picking hyperparameters at random
Sampling at random doesn't mean sampling uniformly at random, over the range of valid values. Instead, it's important to pick the appropriate scale on which to explore the hyperparameters.

Let's say that you're trying to choose the number of hidden units, $n^{[l]}$, for a given layer l. And let's say that you think a good range of values is somewhere from 50 to 100.

In that case, if you look at the number line from 50 to 100, maybe picking some number values at random within this number line. There's a pretty visible way to search for this particular hyperparameter. 

![alt text](_assets/NumHiddenUnit.png)

Or if you're trying to decide on the number of layers in your neural network, we're calling that capital L. Maybe you think the total number of layers should be somewhere between 2 to 4. Then sampling uniformly at random, along 2, 3 and 4, might be reasonable. Or even using a grid search, where you explicitly evaluate the values 2, 3 and 4 might be reasonable.

![alt text](_assets/NumLayer.png)

So these were a couple examples where sampling uniformly at random over the range you're contemplating; might be a reasonable thing to do. But this is not true for all hyperparameters.

### Appropriate scale for hyperparameters
Say your searching for the hyperparameter alpha, the learning rate. And let's say that you suspect 0.0001 might be on the low end, or maybe it could be as high as 1. Now if you draw the number line from 0.0001 to 1, and sample values uniformly at random over this number line. About 90% of the values you sample would be between 0.1 and 1. So you're using 90% of the resources to search between 0.1 and 1, and only 10% of the resources to search between 0.0001 and 0.1. So that doesn't seem right.

![alt text](_assets/LearningRate1.png)

Instead, it seems more reasonable to search for hyperparameters on a log scale. Where instead of using a linear scale, you'd have 0.0001 here, and then 0.001, 0.01, 0.1, and then 1. And you instead sample uniformly, at random, on this type of logarithmic scale.

![alt text](_assets/LearningRate2.png)

Now you have more resources dedicated to searching between 0.0001 and 0.001, and between 0.001 and 0.01, and so on. 

In Python, the way you implement this, is let

```python
r = -4 * np.random.rand()
```

Then a randomly chosen value of alpha, would be

alpha = $10^r$

So after the first line, r will be a random number between -4 and 0. 

And so alpha here will be between $10^{-4}$ and $10^0$. So $10^{-4}$ is this left thing, this $10^{-4}$, and 1 is $10^0$

Generally, if we are trying to sample between $10^a$ to $10^b$, on the log scale, in this example, this is 10 to the a. And you can figure out what a is by taking the log base 10 of 0.0001, which is going to tell you a is -4. And this value on the right, this is 10 to the b. And you can figure out what b is, by taking log base 10 of 1, which tells you b is equal to 0. So what you do, is then sample r uniformly, at random, between a and b. So in this case, r would be between -4 and 0. And you can set alpha, on your randomly sampled hyperparameter value, as 10 to the r.

Just to recap, to sample on the log scale, you take the low value, take logs to figure out what is a. Take the high value, take a log to figure out what is b. So now you're trying to sample, from 10 to the a to the b, on a log scale. So you set r uniformly, at random, between a and b. And then you set the hyperparameter to be 10 to the r. So that's how you implement sampling on this logarithmic scale.

![alt text](_assets/LearningRate3.png)

### Hyperparameters for exponentially weighted averages
Finally, one other tricky case is sampling the hyperparameter beta, used for computing exponentially weighted averages.

So let's say you suspect that beta should be somewhere between 0.9 to 0.999. Maybe this is the range of values you want to search over. So remember, that when computing exponentially weighted averages, using 0.9 is like averaging over the last 10 values. Kind of like taking the average of 10 days temperature, whereas using 0.999 is like averaging over the last 1,000 values. 

![alt text](_assets/beta1.png)

If you want to search between 0.9 and 0.999, it doesn't make sense to sample on the linear scale. Uniformly, at random, between 0.9 and 0.999. So the best way to think about this, is that we want to explore the range of values for 1 minus beta, which is going to now range from 0.1 to 0.001. And so we'll sample the between beta, taking values from 0.1, to maybe 0.1, to 0.001. So using the method we have figured out on the previous slide, this is 10 to the -1, this is 10 to the -3.

Notice on the previous slide, we had the small value on the left, and the large value on the right, but here we have reversed. We have the large value on the left, and the small value on the right.

So what you do, is you sample r uniformly, at random, from -3 to -1. And you set 1- beta = 10 to the r, and so beta = 1- 10 to the r. And this becomes your randomly sampled value of your hyperparameter, chosen on the appropriate scale. And hopefully this makes sense, in that this way, you spend as much resources exploring the range 0.9 to 0.99, as you would exploring 0.99 to 0.999. 

![alt text](_assets/beta2.png)

If you want to study more formal mathematical justification for why we're doing this, right, why is it such a bad idea to sample in a linear scale?

It is that, when beta is close to 1, the sensitivity of the results you get changes, even with very small changes to beta. So if beta goes from 0.9 to 0.9005, it's no big deal, this is hardly any change in your results. But if beta goes from 0.999 to 0.9995, this will have a huge impact on exactly what your algorithm is doing. In both of these cases, it's averaging over roughly 10 values. But here it's gone from an exponentially weighted average over about the last 1,000 examples, to now, the last 2,000 examples. And it's because that formula we have, 1 / 1- beta, this is very sensitive to small changes in beta, when beta is close to 1.

So what this whole sampling process does, is it causes you to sample more densely in the region of when beta is close to 1. Or, alternatively, when 1- beta is close to 0. So that you can be more efficient in terms of how you distribute the samples, to explore the space of possible outcomes more efficiently. 

Think of the β range (0 to 1) as a ruler:

```lua
0.0          0.1          0.2          ...         0.9          1.0
|------------|------------|------------|------------|------------|

```

Uniform sampling gives equal attention to each part of this ruler.
But the important region (0.9 → 1.0) is squeezed into a tiny space at the end.

Now, let’s define 1 − β.
If β = 0.9 → 1−β = 0.1
If β = 0.99 → 1−β = 0.01
If β = 0.999 → 1−β = 0.001

So 1−β can vary over several orders of magnitude (0.1, 0.01, 0.001...).

If you plot that on a logarithmic scale, those values are evenly spaced:

```cpp
log10(0.1) = -1
log10(0.01) = -2
log10(0.001) = -3
```

So, instead of sampling β directly, we sample the exponent uniformly:

```python
r = np.random.uniform(-3, -1)
beta = 1 - 10**r
```

This gives us equal chance of getting 0.9, 0.99, and 0.999 — which are all important, distinct behaviors.

## Hyperparameters Tuning in Practice: Pandas vs. Caviar
### Re-test hyperparameters occasionally
Deep learning today is applied to many different application areas and that intuitions about hyperparameter settings from one application area may or may not transfer to a different one.

There is a lot of cross-fertilization among different applications' domains, so for example, I've seen ideas developed in the computer vision community, such as Confonets or ResNets, which we'll talk about in a later course, successfully applied to speech. I've seen ideas that were first developed in speech successfully applied in NLP, and so on.

So one nice development in deep learning is that people from different application domains do read increasingly research papers from other application domains to look for inspiration for cross-fertilization.

In terms of your settings for the hyperparameters, though, I've seen that intuitions do get stale. So even if you work on just one problem, say logistics, you might have found a good setting for the hyperparameters and kept on developing your algorithm, or maybe seen your data gradually change over the course of several months, or maybe just upgraded servers in your data center. And because of those changes, the best setting of your hyperparameters can get stale. So I recommend maybe just retesting or reevaluating your hyperparameters at least once every several months to make sure that you're still happy with the values you have.

![alt text](_assets/ReTestHyperparameters.png)

Finally, in terms of how people go about searching for hyperparameters, I see maybe two major schools of thought, or maybe two major different ways in which people go about it.

One way is if you babysit one model. And usually you do this if you have maybe a huge data set but not a lot of computational resources, not a lot of CPUs and GPUs, so you can basically afford to train only one model or a very small number of models at a time. In that case you might gradually babysit that model even as it's training.

For example, on Day 0 you might initialize your parameter as random and then start training. And you gradually watch your learning curve, maybe the cost function J or your dataset error or something else, gradually decrease over the first day. Then at the end of day one, you might say, gee, looks it's learning quite well, I'm going to try increasing the learning rate a little bit and see how it does. And then maybe it does better. And then that's your Day 2 performance. And after two days you say, okay, it's still doing quite well. Maybe I'll fill the momentum term a bit or decrease the learning variable a bit now, and then you're now into Day 3. And every day you kind of look at it and try nudging up and down your parameters. And maybe on one day you found your learning rate was too big. So you might go back to the previous day's model, and so on. But you're kind of babysitting the model one day at a time even as it's training over a course of many days or over the course of several different weeks.

![alt text](_assets/BabySittingOneModel.png)

So that's one approach, and people that babysit one model, that is watching performance and patiently nudging the learning rate up or down. But that's usually what happens if you don't have enough computational capacity to train a lot of models at the same time. 

The other approach would be if you train many models in parallel. 

So you might have some setting of the hyperparameters and just let it run by itself, either for a day or even for multiple days, and then you get some learning curve like that; and this could be a plot of the cost function J or cost of your training error or cost of your dataset error, but some metric in your tracking. And then at the same time you might start up a different model with a different setting of the hyperparameters. And so, your second model might generate a different learning curve. I will say that one looks better. And at the same time, you might train a third model, which might generate a learning curve that looks like that, and another one that, maybe this one diverges so it looks like that, and so on. Or you might train many different models in parallel, where these orange lines are different models, right, and so this way you can try a lot of different hyperparameter settings and then just maybe quickly at the end pick the one that works best. Looks like in this example it was, maybe this curve that look best. 

![alt text](_assets/ManyModelInParallel.png)

So to make an analogy, I'm going to call the approach babysitting one model is the panda approach. When pandas have children, they have very few children, usually one child at a time, and then they really put a lot of effort into making sure that the baby panda survives. So that's really babysitting. One model or one baby panda.

Whereas the approach Many Model In Parallel is more like what fish do. I'm going to call this the caviar strategy. There's some fish that lay over 100 million eggs in one mating season. But the way fish reproduce is they lay a lot of eggs and don't pay too much attention to any one of them but just see that hopefully one of them, or maybe a bunch of them, will do well.

![alt text](_assets/Analogy.png)

So I guess, this is really the difference between how mammals reproduce versus how fish and a lot of reptiles reproduce. But I'm going to call it the panda approach versus the caviar approach, since that's more fun and memorable.

The way to choose between these two approaches is really a function of how much computational resources you have. If you have enough computers to train a lot of models in parallel, then by all means take the caviar approach and try a lot of different hyperparameters and see what works.

But in some application domains, I see this in some online advertising settings as well as in some computer vision applications, where there's just so much data and the models you want to train are so big that it's difficult to train a lot of models at the same time. It's really application dependent of course, but I've seen those communities use the panda approach a little bit more, where you are kind of babying a single model along and nudging the parameters up and down and trying to make this one model work. 

Although, of course, even the panda approach, having trained one model and then seen it work or not work, maybe in the second week or the third week, maybe I should initialize a different model and then baby that one along just like even pandas, I guess, can have multiple children in their lifetime, even if they have only one, or a very small number of children, at any one time.

## Normalizing Activations in a Network
### Normalizing inputs to speed up learning
In the rise of deep learning, one of the most important ideas has been an algorithm called batch normalization, created by two researchers, Sergey Ioffe and Christian Szegedy. Batch normalization makes your hyperparameter search problem much easier, makes your neural network much more robust. The choice of hyperparameters is a much bigger range of hyperparameters that work well, and will also enable you to much more easily train even very deep networks. 

When training a model, such as logistic regression, you might remember that normalizing the input features can speed up learnings in compute the means, subtract off the means from your training sets. Compute the variances. The sum of xi squared. 

![alt text](_assets/LogisticRegression.png)

Compute the mean: $\mu = {1 \over m}\Sigma_{i}x^{(i)}$

Subtract off the means from training set: $X = X - \mu$

Compute the variance: $\sigma^2= {1 \over m}\Sigma_{i}x^{(i)2}$. This is an element-wise squaring.

Normalize your data set according to the variances: $X = {X \over {\sigma}}$

We saw in an earlier video how this can turn the contours of your learning problem from something that might be very elongated to something that is more round, and easier for an algorithm like gradient descent to optimize. 

![alt text](_assets/LogisticRegression2.png)

So this works, in terms of normalizing the input feature values to a neural network, alter the regression. 

![alt text](_assets/DeeperNetwork.png)

How about a deeper model? You have not just input features x, but in this layer you have activations $a^{[2]}$, in this layer, you have activations $a^{[2]}$ and so on. So if you want to train the parameters, say $w^{[3]}$, $b^{[3]}$, then wouldn't it be nice if you can normalize the mean and variance of $a^{[2]}$ to make the training of w3, b3 more efficient? 

In the case of logistic regression, we saw how normalizing x1, x2, x3 maybe helps you train w and b more efficiently. So here, the question is, for any hidden layer, can we normalize, the values of a, let's say $a^{[2]}$, in this example but really any hidden layer, so as to train $w^{[3]}$ $b^{[3]}$ faster? Since $a^{[2]}$ is the input to the next layer, that therefore affects your training of $w^{[3]}$ and $b^{[3]}$.

This is what batch norm does, batch normalization, or batch norm for short, does.

Although technically, we'll actually normalize the values of not $a^{[2]}$ but $z^{[2]}$. There are some debates in the deep learning literature about whether you should normalize the value before the activation function, so $z^{[2]}$, or whether you should normalize the value after applying the activation function, $a^{[2]}$. In practice, normalizing $z^{[2]}$ is done much more often. So that's the version I'll present and what I would recommend you use as a default choice. 

![alt text](_assets/DeeperNetwork2.png)

### Implementing Batch Norm
Given some intermediate values, in your neural net. 

Let's say that you have some hidden unit values $z^{[1]}$ up to $z^{[m]}$, and this is really from some hidden layers, so it'd be more accurate to write this as z for some hidden layer i for i equals 1 through m $z^{[l](i)}$. But to reduce writing, I'm going to omit this [l], just to simplify the notation on this line.

Given these values, what you do is compute the mean as follows. Okay, and all this is specific to some layer l, but I'm omitting the [l]. 

$\mu = {1 \over m}\Sigma_{i}z^{(i)}$

And then you compute the variance using pretty much the formula you would expect

$\sigma^2= {1 \over m}\Sigma_{i}(z^{(i)}-\mu)^2$

Then you would take each the zis and normalize it. 

$z^{(i)}_{norm} = {{z^{(2)}-\mu} \over \sqrt{\sigma^2+\epsilon}}$

For numerical stability, we usually add epsilon to the denominator just in case sigma squared turns out to be zero in some estimate. 

Now we've taken these values z and normalized them to have mean 0 and standard unit variance. So every component of z has mean 0 and variance 1. But we don't want the hidden units to always have mean 0 and variance 1. Maybe it makes sense for hidden units to have a different distribution, so what we'll do instead is compute, I'm going to call this z tilde = gamma zi norm + beta. 

$\tilde{z}^{(i)}=\gamma z^{(i)}_{norm} + \beta$

Gamma and beta are learnable parameters of your model. 

We're using gradient descent, or some other algorithm, like the gradient descent of momentum, or RMSProper or Adam, you would update the parameters gamma and beta, just as you would update the weights of your neural network. 

Notice that the effect of gamma and beta is that it allows you to set the mean of $\tilde{z}$ to be whatever you want it to be. In fact, if gamma equals square root sigma squared plus epsilon

$\gamma=\sqrt{\sigma^2 + \epsilon}$

If gamma were equal to this denominator term. And if beta were equal to mu

$\beta = \mu$

Then the effect of $\gamma z^{(i)}_{norm} + \beta$ is that it would exactly invert this equation $z^{(i)}_{norm}$.

If

$\gamma=\sqrt{\sigma^2 + \epsilon}$ 

and

$\beta = \mu$

Then

$\tilde{z}=z^{(i)}$

By an appropriate setting of the parameters gamma and beta, this normalization step, that is, these above four equations is just computing essentially the identity function. But by choosing other values of gamma and beta, this allows you to make the hidden unit values have other means and variances as well. 

The way you fit this into your neural network is, whereas previously you were using these values $z^{(1)}$, $z^{(2)}$, and so on, you would now use $\tilde{z}^{(i)}$, instead of $z^{(i)}$ for the later computations in your neural network. And you want to put back in this [l] to explicitly denote which layer it is in, you can put it back there $z^{[l](i)}$, $\tilde{z}^{[l](i)}$.

The intuition I hope you'll take away from this is that we saw how normalizing the input features X can help learning in a neural network. And what batch norm does is it applies that normalization process not just to the input layer, but to the values even deep in some hidden layer in the neural network. So it will apply this type of normalization to normalize the mean and variance of some of your hidden units' values, z.

One difference between the training input and these hidden unit values is you might not want your hidden unit values be forced to have mean 0 and variance 1.

For example, if you have a sigmoid activation function, you don't want your values to always be clustered here. 

![alt text](_assets/Sigmoid.png)

You might want them to have a larger variance or have a mean that's different than 0, in order to better take advantage of the nonlinearity of the sigmoid function rather than have all your values be in just this linear regime. 

![alt text](_assets/LinearRegime.png)

So that's why with the parameters gamma and beta, you can now make sure that your $z^{(i)}$ values have the range of values that you want. 

What it does really is it then shows that your hidden units have standardized mean and variance, where the mean and variance are controlled by two explicit parameters gamma and beta which the learning algorithm can set to whatever it wants. So what it really does is it normalizes in mean and variance of these hidden unit values, really the $z^{(i)}$, to have some fixed mean and variance. And that mean and variance could be 0 and 1, or it could be some other value, and it's controlled by these parameters gamma and beta.

### Explained by ChatGPT
You already know that before training a model, it helps to normalize the input features (e.g., make each feature have mean 0 and variance 1). This helps gradient descent converge faster because all features are on a similar scale.

But now imagine a deep network — not just the input layer, but many hidden layers. Even if your input is normalized, the hidden layer activations can get out of control (too big or too small) during training.

As training goes on, the distribution of activations (Z or A) inside the network keeps changing because earlier layers keep updating their weights.

So, the next layer keeps receiving inputs with different scales — sometimes very large, sometimes very small.

That means:
* Gradients may explode or vanish.
* Training slows down.
* The network becomes harder to tune.

This constant change of distributions inside the network is called Internal Covariate Shift.

Batch Normalization says:

“Why don’t we normalize the activations inside the network, just like we normalize inputs?”

So, for every layer (say layer l), during training, we take all activations $Z^{[l]}$ from the mini-batch and normalize them.

After normalization, we still want the network to be flexible, so we add two learnable parameters — γ (gamma) and β (beta).

$\tilde{z}^{(i)}=\gamma z^{(i)}_{norm} + \beta$

This lets the network “undo” normalization if it’s better for performance.

For example:
* If the best activations should be around 5, not 0 → β shifts the mean.
* If they should be more spread out → γ scales the variance.

Imagine each layer in your neural network is like a worker on a factory line.
If one worker keeps sending different-sized boxes every few minutes, the next worker has to constantly readjust their tools.

Batch Normalization is like standardizing the box size before passing it on — so every worker (layer) can do their job smoothly without constant adjustments.

Imagine activations (Z values) before normalization:

Distribution 1: mean = 100, variance = 500
Distribution 2: mean = -3, variance = 0.1

After BatchNorm:

Distribution (normalized): mean = 0, variance = 1

No matter how weird the raw Z values are, the network now works on a stable range of values, layer by layer.

Suppose you have a bunch of numbers — say the activations (outputs) from some neurons in a layer:

[2, 4, 6, 8, 10]

Step 1. Compute the mean (average)

mean = (2+4+6+8+10)/5 = 6

Step 2. Compute the variance

Variance measures how spread out the numbers are from their mean.

variance = (2−6)^2+(4−6)^2+(6−6)^2+(8−6)^2+(10−6)^2​/5 = 8

The higher the variance, the more spread out your data is.

We normalize by subtracting the mean (so data is centered around 0)
and dividing by the standard deviation (so it has a consistent spread).

Step 3. Subtract the mean (make mean = 0)

Subtract 6 from each number:

[2-6, 4-6, 6-6, 8-6, 10-6] = [-4, -2, 0, 2, 4]

Now if you take the average:

mean = (-4-2+0+2+4)/5 = 0

So now the data is centered around zero.

Step 4. Divide by the standard deviation (make variance = 1)

Standard deviation = √variance = √8 ≈ 2.83

Divide each number by 2.83:

[-4/2.83, -2/2.83, 0, 2/2.83, 4/2.83] ≈ [-1.41, -0.71, 0, 0.71, 1.41]


Now:
* Mean ≈ 0
* Variance ≈ 1

We’ve forced mean = 0, variance = 1

Why do this?

Imagine you’re training a neural network.

If activations (numbers flowing through the layers) are too big or too small, the gradients can explode or vanish — making learning unstable and slow.

By “forcing” mean = 0 and variance = 1 at every layer:
* Activations are centered (half positive, half negative)
* They have a consistent scale (not too wide or narrow)
* Gradients are stable → training becomes much smoother

## Fitting Batch Norm into a Neural Network
### Adding Batch Norm to a network
You have seen the equations for how to invent Batch Norm for maybe a single hidden layer. Let's see how it feeds into the training of a deep network. 

Let's say you have a neural network like this.

![alt text](_assets/NN.png)

You've seen me say before that you can view each of the unit as computing two things. First, it computes Z and then it applies the activation function to compute A. So we can think of each of these circles as representing a two-step computation.

![alt text](_assets/NN2.png)

If you were not applying Batch Norm, you would have an input X feed into the first hidden layer, and then first compute $Z^{[1]}$, and this is governed by the parameters $W^{[1]}$ and $b^{[1]}$. Then feed $Z^{[1]}$ into the activation function to compute $a^{[1]}$.

But what would do in Batch Norm is take this value $Z^{[1]}$, and apply Batch Norm, sometimes abbreviated BN to it, and that's going to be governed by parameters, Beta 1 and Gamma 1, and this will give you this new normalize value $Z^{[1]}$. And then you feed that to the activation function to get $a^{[1]}$, which is $g^{[1]}$ applied to $\tilde{Z}^{[1]}$.

![alt text](_assets/NormalizeFirstLayer.png)

Now, you've done the computation for the first layer, where this Batch Norms that really occurs in between the computation from Z and a.

Next, you take this value $a^{[1]}$ and use it to compute $Z^{[2]}$, and so this is now governed by $W^{[2]}$ and $b^{[2]}$. And similar to what you did for the first layer, you would take $Z^{[2]} and apply it through Batch Norm, and we abbreviate it to BN now. This is governed by Batch Norm parameters specific to the next layer. So Beta 2, Gamma 2, and now this gives you $\tilde{Z}^{[2]}$, and you use that to compute $a^{[2]}$ by applying the activation function, and so on.

![alt text](_assets/BatchNorm.png)

So once again, the Batch Norms that happens between computing Z and computing a. 

The intuition is that, instead of using the un-normalized value Z, you can use the normalized value $\tilde{Z}$, that's the first layer. The second layer as well, instead of using the un-normalized value $Z^{[2]}$, you can use the mean and variance normalized values $\tilde{Z}^{[2]}$. 

The parameters of your network are going to be $W^{[1]}$ and $b^{[1]}$. It turns out we'll get rid of the parameters but we'll see why in the next slide. But for now, imagine the parameters are the usual $W^{[1]}$, $b^{[1]}$ to $W^{[L]}$, $b^{[L]}$. We added to this new network additional parameters $\beta^{[1]}$, $\gamma^{[1]}$ to $\beta^{[L]}$, $\gamma^{[L]}$. For each layer in which you are applying Batch Norm. 

For clarity, note that these Betas here, these have nothing to do with the hyperparameter beta that we had for momentum over the computing the various exponentially weighted averages. The authors of the Adam paper use Beta on their paper to denote that hyperparameter, the authors of the Batch Norm paper had used Beta to denote this parameter, but these are two completely different Betas. I decided to stick with Beta in both cases, in case you read the original papers. But the Beta 1, Beta 2, and so on, that Batch Norm tries to learn is a different Beta than the hyperparameter Beta used in momentum and the Adam and RMSprop algorithms.

These are the new parameters of your algorithm, you would then use whether optimization you want, such as gradient descent in order to implement it.

![alt text](_assets/Parameters.png)

For example, you might compute $d\beta^{[l]}$ a given layer, and then update the parameters Beta, gets updated as $\beta=\beta-\alpha d\beta^{[l]}$. And you can also use Adam or RMSprop or momentum in order to update the parameters Beta and Gamma, not just gradient descent.

And even though in the previous video, I had explained what the Batch Norm operation does, computes mean and variances and subtracts and divides by them. If they are using a Deep Learning Programming Framework, usually you won't have to implement the Batch Norm step on Batch Norm layer yourself. So the probing frameworks, that can be sub one line of code. So for example, in terms of flow framework, you can implement Batch Normalization with this function. 

tf.nn.batch_normalization

We'll talk more about probing frameworks later, but in practice you might not end up needing to implement all these details yourself, knowing how it works so that you can get a better understanding of what your code is doing. But implementing Batch Norm is often one line of code in the deep learning frameworks.

So far, we've talked about Batch Norm as if you were training on your entire training site at the time as if you are using Batch gradient descent. In practice, Batch Norm is usually applied with mini-batches of your training set.

### Working with mini-batches
The way you actually apply Batch Norm is you take your first mini-batch and compute $Z^{[1]}$. Same as we did on the previous slide using the parameters $W^{[1]}$, $b^{[1]}$ and then you take just this mini-batch and computer mean and variance of the $Z^{[1]}$ on just this mini batch and then Batch Norm would subtract by the mean and divide by the standard deviation and then re-scale by Beta 1, Gamma 1, to give you $\tilde{Z}^{[1]}$, and all this is on the first mini-batch, then you apply the activation function to get $a^{[1]}$, and then you compute $Z^{[2]}$ using $W^{[2]}$, $b^{[2]}$, and so on.

![alt text](_assets/FirstMiniBatch.png)

So you do all this in order to perform one step of gradient descent on the first mini-batch and then goes to the second mini-batch $X^{\{2\}}$, and you do something similar where you will now compute $Z^{[1]}$ on the second mini-batch and then use Batch Norm to compute $\tilde{Z}^{[1]}$. And so here in this Batch Norm step, you would be normalizing $\tilde{Z}^{[1]}$ using just the data in your second mini-batch, so does Batch Norm step here. Let's look at the examples in your second mini-batch, computing the mean and variances of the $Z^{[1]}$'s on just that mini-batch and re-scaling by Beta and Gamma to get $\tilde{Z}^{[1]}$, and so on. And you do this with a third mini-batch, and keep training. 

![alt text](_assets/SecondMiniBatch.png)

Parameters: $W^{[l]}$, $b^{[l]}$, $\beta^{[l]}$, $\gamma^{[l]}$

$Z^{[l]} = W^{[l]}a^{[l-1]}+b^{[l]}$

What Batch Norm does, is it is going to look at the mini-batch and normalize $Z^{[l]}$ to first of mean 0 and standard variance, and then a rescale by Beta and Gamma. But what that means is that, whatever is the value of $b^{[l]}$ is actually going to just get subtracted out, because during that Batch Normalization step, you are going to compute the means of the $Z^{[l]}$'s and subtract the mean. And so adding any constant to all of the examples in the mini-batch, it doesn't change anything. Because any constant you add will get cancelled out by the mean subtractions step. So, if you're using Batch Norm, you can actually eliminate that parameter, or if you want, think of it as setting it permanently to 0.

Parameterization becomes: $Z^{[l]} = W^{[l]}a^{[l-1]}$. Then compute $^{[l]}_{norm}$, then compute $\tilde{Z}^{[l]}=\gamma Z^{[l]}+\beta^{[l]}$. You end up using this parameter $\beta^{[l]}$ in order to decide whats that mean of $\tilde{Z}^{[l]}$. Which is why guess post in this layer.

Just to recap, because Batch Norm zeroes out the mean of these $Z^{[l]}$ values in the layer, there's no point having this parameter $b^{[l]}$, and so you must get rid of it, and instead is sort of replaced by $\beta^{[l]}$, which is a parameter that controls that ends up affecting the shift or the biased terms.

Finally, remember that the dimension of $Z^{[l]}$, because if you're doing this on one example, it's going to be ($n^{[l]}$,1), and so $b^{[l]}$, a dimension, ($n^{[l]}$,1), if $n^{[l]}$ was the number of hidden units in layer L. And so the dimension of $\beta^{[l]}$ and $\gamma^{[l]}$ is also going to be $n^{[l]}$ by 1 because that's the number of hidden units you have. You have $n^{[l]}$ hidden units, and so $\beta^{[l]}$ and $\gamma^{[l]}$ are used to scale the mean and variance of each of the hidden units to whatever the network wants to set them to.

### Implementing gradient descent
Assuming you're using mini-batch gradient descent

for t=1 to number of Mini batches:
> Compute forward prop on mini batch $X^{\{t\}}$ \
>> In each hidden layer, use Batch Norm to replace $Z^{[l]}$ with $\tilde{Z}^{[l]}$, it ensures that within that mini-batch, the value Z end up with some normalized mean and variance and the values and the version of the normalized mean that and variance is $\tilde{Z}^{[l]}$. \
> Use backprop to compute $dW^{[l]}$, $db^{[l]}$, $d\beta^{[l]}$, $d\gamma^{[l]}$. Although, technically, since you have got to get rid of b, $db^{[l]}$ actually now goes away. \
> Finally, update the parameters. $W^{[l]}=W-\alpha dW^{[l]}$, $\beta^{[l]}=\beta^{[l]}-\alpha d\beta^{[l]}$, so on

If you have computed the gradient as follows, you could use gradient descent. That's what I've written here, but this also works with gradient descent with momentum, or RMSprop, or Adam. Where instead of taking this gradient descent update,nini-batch you could use the updates given by these other algorithms.

As $Z^{[l]} = W^{[l]}a^{[l-1]}+b^{[l]}$

Then BatchNorm subtracts the mean:

$Z^{[l]}_{norm}={{(W^{[l]}a^{[l-1]}+b^{[l]})-\mu^{[l]}} \over {\sqrt{\sigma^{2[l]} + \epsilon}}}$

But notice this: $\mu^{[l]}$ is computed from all values in the batch of $Z^{[l]}$ which includes the effect of $b^{[l]}$. Whatever shift $b^{[l]}$ tries to introduce, BatchNorm immediately subtracts it out by removing the batch mean. So effectively, β replaces the job of the bias.

![alt text](_assets/Step1.png)

![alt text](_assets/Step2.png)

![alt text](_assets/Step3.png)

![alt text](_assets/Step4.png)



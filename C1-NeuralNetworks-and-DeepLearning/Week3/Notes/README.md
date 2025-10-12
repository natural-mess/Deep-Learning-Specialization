# Shallow Neural Networks
Build a neural network with one hidden layer, using forward propagation and backpropagation.

**Learning Objectives**
* Describe hidden units and hidden layers
* Use units with a non-linear activation function, such as tanh
* Implement forward and backward propagation
* Apply random initialization to your neural network
* Increase fluency in Deep Learning notations and Neural Network Representations
* Implement a 2-class classification neural network with a single hidden layer
* Compute the cross entropy loss

## Neural Networks Overview
### What is a Neural Network ?
In Logistic Regression, we put the features x and parameters w and b that allows you to compute z which is then used to computes a, and we were using a interchangeably with this output y hat and then you can compute the loss function, L.

![alt text](_assets/LogisticRegression.png)

![alt text](_assets/LogisticRegression2.png)

Neural network

![alt text](_assets/NeuralNetwork.png)

We can form a neural network by stacking together a lot of little sigmoid units. Previously, the Logistic Regression node corresponds to 2 steps to calculate. 
1. Compute the z-value
2. Computes a-value

In Neural Network, a node correspond to a z-like calculation and a-like calculation like in Logistic Regression.

We use superscript [1] to refer to calculation of nodes in 1st layer, and superscript [2] to refer to calculation of nodes in 2nd layer.

$z^{[1]} = W^{[1]}*x+b^{[1]}$

Superscript (1) is used to refer to training examples. So $X^{(1)}$ refers to i-th training example.

After computing $z^{[1]}$, we need to compute $a^{[1]}$, then we compute $z^{[2]} = W^{[2]}*x+b^{[2]}$, then compute $a^{[2]}$. Then use $a^{[2]}$ to compute $\ell(a^{[2]},y)$.

Then perform backward propagation to compute w and b.

![alt text](_assets/WhatIsNN.png)

## Neural Network Representation
Signal hidden layer neural network

![alt text](_assets/NeuralNetwork1layer.png)

Input features x1, x2, x3 stacked up vertically. This is called the input layer of the neural network. The next layer is called the hidden layer. The final layer is output layer, which is responsible for generating predicted value $\hat{y}$

![alt text](_assets/NNlayers.png)

The term "hidden layer" refers to the fact that in the training set, the true values for these nodes in the middle are not observed, that is, you don't see what they should be in the training set. You see what the inputs are, you see what the output should be, but the things in the hidden layer are not seen in the training set.

Whereas previously, we were using the vector `X` to denote the input features and alternative notation for the values of the input features will be $a^{(0)}$. And the term `a` also stands for activations, and it refers to the values that different layers of the neural network are passing on to the subsequent layers.

Input layer passes on the value `X` to the hidden layer, so we call that activations of the input layer $a^{[0]}$.

Hidden layer generates some sort of activation $a^{[1]}$. First node generates $a^{[1]}_1$, 2nd node generate $a^{[1]}_2$, and so on.

So $a^{[1]}$ is a 4 dimensional vector or 4x1 matrix, or a 4 column vector.

![alt text](_assets/a_superscript_1.png)

Then finally, the output layer regenerates some value $a^{[2]}$, which is just a real number. And so $\hat{y}$ is going to take on the value of $a^{[2]}$. So this is analogous to how in logistic regression we have $\hat{y}$ equals a and in logistic regression which we only had that one output layer, so we don't use the superscript square brackets. But with our neural network, we now going to use the superscript square bracket to explicitly indicate which layer it came from. 

![alt text](_assets/NN2layer.png)

This network is called 2 layer neural network. Because when we count layers in NN, we don't count the input layer.
- Input layer = layer 0
- Hidden layer = layer 1
- Output layer = layer 2

The hidden layer and the output layers will have parameters associated with them. 
- The hidden layer will have parameters $w^{[1]}$ and $b{[1]}$, in this case w is 4x3 matrix (4 nodes hidden layer and 3 input features) and b is 4x1 vector. 
- Output layer will have $w^{[2]}$ which is 1x4 vector (hidden layer has 4 hidden units and output layer has just 1 unit) and $b{[2]}$ which is 1x1.

## Computing a Neural Network's Output
We've said before that logistic regression, the circle in logistic regression, really represents two steps of computation rows: compute z and a (activation as a sigmoid function of z).

![alt text](_assets/LRNode.png)

So a neural network just does this a lot more times.

![alt text](_assets/NeuralNetwork1.png)

Similar to Logistic Regression, first node of hidden layer does 2 steps of computation: $z^{[1]}_1$ and $a^{[1]}_1$
* Step 1: $z^{[1]}_1 = w^{[1]T}_1x + b^{[1]}_1$
* Step 2: $a^{[1]}_1 = \sigma(z^{[1]}_1)$

![alt text](_assets/hiddenLayer1stNode.png)

Second node of hidden layer also does 2 steps of computation: $z^{[1]}_2$ and $a^{[1]}_2$
* Step 1: $z^{[1]}_2 = w^{[1]T}_2x + b^{[1]}_2$
* Step 2: $a^{[1]}_2 = \sigma(z^{[1]}_2)$

![alt text](_assets/hiddenLayerComputation.png)

Same for hidden layer unit 3 and 4.

![alt text](_assets/NeuralNetwork1layer.png)

* $z^{[1]}_1 = w^{[1]T}_1x + b^{[1]}_1$, $a^{[1]}_1 = \sigma(z^{[1]}_1)$
* $z^{[1]}_2 = w^{[1]T}_2x + b^{[1]}_2$, $a^{[1]}_2 = \sigma(z^{[1]}_2)$
* $z^{[1]}_3 = w^{[1]T}_3x + b^{[1]}_3$, $a^{[1]}_3 = \sigma(z^{[1]}_3)$
* $z^{[1]}_4 = w^{[1]T}_4x + b^{[1]}_4$, $a^{[1]}_4 = \sigma(z^{[1]}_4)$

Using for loop to calculate above computations is inefficient.

-> Use vectorization.

Take $w_1$ to $w_4$ and stack them into a 4x3 matrix. Note that $w^{[1]T}_1$ is row vector.

![alt text](_assets/w_transpose.png)

Then we take this matrix and multiply it with input features x1, x2, x3. We have

![alt text](_assets/wx.png)

Then add b to it.

![alt text](_assets/wxPlusb.png)

Denote this as $z^{[1]}$

![alt text](_assets/z1.png)

To compute $a^{[1]}$, we take together $a^{[1]}_1$ to $a^{[1]}_4$, this will be equal to $\sigma(z^{[1]})$

![alt text](_assets/a1.png)

![alt text](_assets/NeuralNetworkCompute.png)

Recap: Given input x:
* $z^{[1]} = w^{[1]T}x + b^{[1]}$
* $a^{[1]} = \sigma(z^{[1]})$

Remember that x = $a^{[0]}$ and y = $a^{[2]}$, so we can represent z as

$z^{[1]} = w^{[1]T}a^{[0]} + b^{[1]}$

Similar for second layer
* $z^{[2]} = w^{[2]T}x + b^{[2]}$
* $a^{[2]} = \sigma(z^{[2]})$

![alt text](_assets/NeuralNetworkCompute2.png)

If you think of the upper unit as just being analogous to logistic regression which have parameters w and b, w really plays an analogous role to $w^{[2]}$ transpose, or $w^{[2]}$ is really W transpose and b is equal to $b^{[2]}$. 

## Vectorizing Across Multiple Examples

![alt text](_assets/NeuralNetworkCompute3.png)

This tells us, given an input feature x, we can use them to generate $a^{[2]} = \hat{y}$ for a single training example.

If we have m training examples, we need t repeat this process for say:
* $x^{(1)}$ to compute $a^{[2](1)}=\hat{y}^{[1]}$ (does a prediction on 1st training example)
* $x^{(2)}$ to compute $a^{[2](2)}=\hat{y}^{[2]}$
* So on
* $x^{(m)}$ to compute $a^{[2](m)}=\hat{y}^{[m]}$

Notation $a^{[2](i)}$, the (i) refers to training example i-th, and [2] refers to layer 2.

If you have an unvectorized implementation and want to compute the predictions of all your training examples, you need to do 

> for i = 1 to m:
>> $z^{[1](i)} = w^{[1]}x^{(i)} + b^{[1]}$ \
>> $a^{[1](i)} = \sigma(z^{[1](i)})$ \
>> $z^{[2](i)} =  w^{[2]}x^{(i)} + b^{[2]}$ \
>> $a^{[2](i)} = \sigma(z^{[2](i)})$

-> Vectorize these equations to get rid of this for loop.

![alt text](_assets/VectorizingNN.png)

Recall that we defined the matrix X to be equal to our training examples stacked up in columns

![alt text](_assets/StackX.png)

$Z^{[1]} = W^{[1]}X + b^{[1]}$ \
$A^{[1]} = \sigma(Z^{[1]})$ \
$Z^{[2]} =  W^{[2]}W + b^{[2]}$ \
$A^{[2]} = \sigma(Z^{[2]})$ 

We went from lower case x to capital case X by stacking up lower case x's in different columns. If we do the same thing for z and a, we have

![alt text](_assets/StackzAnda.png)

One of the property of this notation that might help you to think about it is that this matrixes say Z and A, horizontally we're going to index across training examples. The horizontal index corresponds to different training example, when you sweep from left to right you're scanning through the training set. And vertically this vertical index corresponds to different nodes in the neural network. So for example, the value at the top most, top left most corner of the mean corresponds to the activation of the first heading unit on the first training example. One value down corresponds to the activation in the second hidden unit on the first training example, then the third heading unit on the first training sample and so on.

So as you scan down this is your indexing to the hidden units number. Whereas if you move horizontally, then you're going from the first hidden unit. And the first training example to now the first hidden unit and the second training sample, the third training example. So on until the node corresponds to the activation of the first hidden unit on the final training example and the m-th training example. 

## Explanation for Vectorized Implementation
For 1st training example:

$z^{[1](1)} = w^{[1]}x^{(1)} + b^{[1]}$

For 2nd training example:

$z^{[1](2)} = w^{[1]}x^{(2)} + b^{[1]}$

For 3rd training example:

$z^{[1](3)} = w^{[1]}x^{(3)} + b^{[1]}$

To simplify this justification a little bit that b is equal to zero. But the argument we're going to lay out will work with just a little bit of a change even when b is non-zero.

![alt text](_assets/VectorizationJustification.png)

We have $w^{[1]}$ is a matrix.

![alt text](_assets/w1.png)

$w^{[1]}$ times $x^{(1)}$ gives some column vector

![alt text](_assets/w1x1.png)

$w^{[1]}$ times $x^{(2)}$ gives some other column vector

![alt text](_assets/w1x2.png)

$w^{[1]}$ times $x^{(3)}$ gives some other column vector

![alt text](_assets/w1x3.png)

If you consider the training set capital X, which we form by stacking together all of our training examples. So the matrix capital X is formed by taking the vector x1 and stacking it vertically with x2 and then also x3. This is if we have only three training examples. If you have more, they'll keep stacking horizontally like that.

![alt text](_assets/matrixX.png)

If we take $W^{[1]}$ and multiply it by matrix X

![alt text](_assets/W1X.png)

With Python broadcasting, you end up having $b^{[i]}$ individually to each of the columns of this matrix.

![alt text](_assets/RecapVectorizing.png)

Here we have two-layer neural network where we go to a much deeper neural network in next week's videos, you see that even deeper neural networks are basically taking these two steps and just doing them even more times than you're seeing here.

## Activation Functions
When you build your neural network, one of the choices you get to make is what activation function to use in the hidden layers as well as at the output units of your neural network. So far, we've just been using the sigmoid activation function, but sometimes other choices can work much better. 

![alt text](_assets/ForwardPropagation.png)

![alt text](_assets/sigmoid.png)

$a = {1 \over {1+e^{-z}}}$

So in the more general case, we can have a different function $g(z^{[1]})$. Where g can be a nonlinear function that may not be the sigmoid function.

$a^{[1]} = g(z^{[1]})$

$a^{[2]} = g(z^{[2]})$

For example, the sigmoid function goes between 0 and 1. An activation function that almost always works better than the sigmoid function is the tangent function or the hyperbolic tangent function.

![alt text](_assets/tangentFunction.png)

$a = tanh(z) = {{e^z - e^{-z} \over {e^z + e^{-z}}}}$

And it's actually mathematically a shifted version of the sigmoid function. 

This almost always works better than the sigmoid function because with values between +1 and -1, the mean of the activations that come out of your hidden layer are closer to having a 0 mean. And so just as sometimes when you train a learning algorithm, you might center the data and have your data have 0 mean using a tanh instead of a sigmoid function. Kind of has the effect of centering your data so that the mean of your data is close to 0 rather than maybe 0.5. And this actually makes learning for the next layer a little bit easier.

_I pretty much never use the sigmoid activation function anymore. The tanh function is almost always strictly superior. The one exception is for the output layer because if y is either 0 or 1, then it makes sense for y hat to be a number that you want to output that's between 0 and 1 rather than between -1 and 1. So the one exception where I would use the sigmoid activation function is when you're using binary classification. In which case you might use the sigmoid activation function for the output layer._

What you see in this example is where you might have a tanh activation function for the hidden layer and sigmoid for the output layer. So the activation functions can be different for different layers. And sometimes to denote that the activation functions are different for different layers, we might use square brackets superscripts as well to indicate that $g^{[1]}$ may be different than $g^{[2]}$.

One of the downsides of both the sigmoid function and the tan h function is that if z is either very large or very small, then the gradient of the derivative of the slope of this function becomes very small. So if z is very large or z is very small, the slope of the function either ends up being close to zero and so this can slow down gradient descent. So one other choice that is very popular in machine learning is what's called the rectified linear unit (ReLU).

|z|	sigmoid(z)| (slope)|
|-|-|-|
|−10|	0.00005	|almost 0|
|0|	0.5|	0.25 (maximum slope)|
|+10|	0.99995|	almost 0|

When z is large positive or large negative, the sigmoid curve flattens out — it’s nearly horizontal.

That means:
* The slope (derivative) = almost 0
* And the gradient in gradient descent = very small

Gradient descent updates a parameter w like this:

$w = w - \alpha * {dJ \over dw}$

* If the gradient is big, w moves a lot.
* If the gradient is tiny, w barely moves.

Consider a simple 3 layer feedforward network (input -> hidden1 -> hidden 2 -> output).

![alt text](_assets/gradientOfLoss.png)

Sigmoid $\sigma(z)$. Derivative of sigmoid is $\sigma(z)(1-\sigma(z))$. Max value is 0.25 when $\sigma = 0.5$. So if $\sigma(z)$ is near 0 or 1 (saturation), its derivative is very small.

Tanh(z), it's derivative is $1-tanh^2(z)$. Max value is 1. If |tanh(z)| is close to 1 (saturation), derivative is small.

Because the overall gradient is a product of many activation derivatives, if each one is small (<1), their product becomes very small.

Assume a deep network with 3 activation derivatives at hidden layers (values representative of saturated sigmoids):
* activation' at layer1 = 0.01
* activation' at layer2 = 0.01
* activation' at output = 0.01
* other multiplicative factors (weights, error term) ≈ 1 for simplicity
Then gradient for an early weight is approximately:

${dL \over {dw_1}} = 1 * 0.01 * 1 * 0.01 * 1 * 0.01 * 1 = 10^{-6}$

If learning rate is 0.1, the update is

$\alpha * {dL \over {dw_1}} ~ 0.1 * 10^{-6} = 10^{-7}$

That update is tiny -> the weight bearely changes -> learning stalls in early layers.

For ReLU
* For positive z, derivative = 1, not a small fraction. So factors in the gradient product that come from ReLU don’t shrink the gradient.
* Thus, in the backward pass, whenever pre-activation is positive, the activation' term is 1 and does not reduce the gradient magnitude.
* If layer derivatives are [1, 1, 0.01] instead of [0.01, 0.01, 0.01], product = 1 x 1 x 0.01 = 0.01 (much larger than $10^{-6}$).

* So ReLU improves the backward pass by keeping many activation derivatives equal to 1 (no shrinking), allowing useful gradients to reach early layers and thus enabling faster learning.
* Sigmoid/tanh squash big inputs to near-constant outputs. When that happens, they become almost flat — the slope is nearly zero.
* Backpropagation has to multiply a bunch of these slopes (one per layer). If many slopes are near zero, the product is near zero. That means the “message” telling early layers how to change gets lost (vanishes).
* ReLU doesn’t squash positive inputs; it has slope 1 for positive inputs. So the “message” (gradient) doesn’t get shrunk as it travels backward through layers — early layers still get a meaningful signal to learn.


![alt text](_assets/ReLU.png)

a = max(0,z)

The derivative is 1 as long as z is positive and derivative or the slope is 0 when z is negative. If you're implementing this, technically the derivative when z is exactly 0 is not well defined. But when you implement this in the computer, the odds that you get exactly z equals 000000000000 is very small. So you don't need to worry about it. In practice, you could pretend a derivative when z is equal to 0, you can pretend is either 1 or 0. And you can work just fine. 

If you're using binary classification, then the sigmoid activation function is very natural choice for the output layer. And then for all other units, ReLU or the rectified linear unit is increasingly the default choice of activation function. So if you're not sure what to use for your hidden layer, I would just use the ReLU activation function, is what you see most people using these days. Although sometimes people also use the tan h activation function. One disadvantage of the ReLU is that the derivative is equal to 0 when z is negative. In practice this works just fine. But there is another version of the ReLU called the Leaky ReLU.

In practice, using the ReLU activation function, your neural network will often learn much faster than when using the tan h or the sigmoid activation function. And the main reason is that there's less of this effect of the slope of the function going to 0, which slows down learning. And I know that for half of the range of z, the slope for ReLU is 0. But in practice, enough of your hidden units will have z greater than 0. So learning can still be quite fast for most training examples. 

![alt text](_assets/Sigmoid1.png)
* Never use this except for the output layer if you are doing binary classification or maybe almost never use this
* Tan h is pretty much strictly superior.

![alt text](_assets/tanh.png)

![alt text](_assets/ReLU1.png)
* Most commonly used activation function
* Not sure what to use, use this

![alt text](_assets/LeakyReLU.png)
* Feel free also to try the Leaky ReLU 
* a=max(0.01z, z)
* 0.01 can be replaced by other parameter.

## Why do you need Non-Linear Activation Functions?


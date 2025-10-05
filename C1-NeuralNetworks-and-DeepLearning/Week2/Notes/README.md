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





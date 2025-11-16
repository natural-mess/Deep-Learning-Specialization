# Week 3: Object Detection

**Learning Objectives**
* Identify the components used for object detection (landmark, anchor, bounding box, grid, ...) and their purpose
* Implement object detection
* Implement non-max suppression to increase accuracy
* Implement intersection over union
* Handle bounding boxes, a type of image annotation popular in deep learning
* Apply sparse categorical crossentropy for pixelwise prediction
* Implement semantic image segmentation on the CARLA self-driving car dataset
* Explain the difference between a regular CNN and a U-net
* Build a U-Net

## Object Localization
### What are localization and detection?
* Image classification: Algorithm looks at the picture and says if it's a car or not car.
* Classification with localization: Algorithm labels the picture as a car and puts a bounding box around the possition of the car.
  * The term "localization" means where is the object in the picture.
* Detection: There are multiple objects in the picture, algorithm has to detect all objects and localize all of them.

Image classification and Classification with localization uslally have 1 object.

Detection can have multiple objects and maybe multiple objects of different categories within a single image.

![alt text](_assets/ObjectLocalization.png)

### Classification with localization
Image classification problem:
* Input: picture
* ConvNet with multiple layers
* Output: vector features is fed to a softmax uni that outputs the predicted class.

For example: Self-driving car, the object categories are:
* Pedesctrian
* Car
* Motocycle
* Background (does not contain 3 objects above)

For localization, change the NN above to have addition outputs which are bounding box parameters.
* bx, by: bounding-box center (normalized)
* bh, bw: bounding-box height and width (normalized)

![alt text](_assets/Localization.png)

### Defining the target label y
Classes:
1. Pedesctrian
2. Car
3. Motocycle
4. Background (does not contain 3 objects above)

Need to output bx, by, bh, bw, class label (1-4).

y = [pc, bx, by, bh, bw, c1, c2, c3]

* pc: probability that there is 1 of the classes we are trying to detect.
  * If it's the background class: pc = 0.
  * Else, pc = 1
    * When pc = 1:
      * Output bx, by, bh, bw to localize object and specify bounding box.
      * Output c1, c2, c3 which tell class 1, class 2 or class 3.

For example:

Input X is an image of a car

```
y = [1, bx, by, bh, bw, 0, 1, 0]
```

Input X is an image of background (no objects)

```
y = [0, ?, ?, ?, ?, ?, ?, ?]
```

"?" means "don't care" because there is no object in the image, then we don't care about bounding box and object classes.

### Loss function to train NN
* y: ground truth label
* $\hat{y}$: NN output

1. If $y_1$ == 1 (or pc == 1, means there is an object in the image)

$\ell(\hat{y}, y) = (\hat{y}_1-y_1)^2 + (\hat{y}_2-y_2)^2 + ... + (\hat{y}_8-y_8)^2$

2. If $y_1$ == 0 (or pc == 0)

$\ell(\hat{y}, y) = (\hat{y}_1-y_1)^2$

Because in the 2nd case, when pc = 0, the rest of output y is "don't care".

![alt text](_assets/Label_y.png)

*Just as a side comment for those of you that want to know all the details, I've used the squared error just to simplify the description here.*
*In practice you could probably use*
* *log like feature loss for the c1, c2, c3 to the softmax output.*
* *One of those elements usually you can use squared error or something like squared error for the bounding box coordinates.*
* *For pc you could use something like the logistics regression loss.*

*Even if we use squared error, it will work okay.*

## Landmark Detection
Landmark detection means: Finding key points (important locations) inside an image.

Examples:
* Eye position in a face
* Nose position
* Mouth corners
* or any body joints (elbows, knees, shoulders)

Instead of saying what is in the image (classification), we want to find where something is.

For each landmark, the model predicts an (x, y) coordinate inside the image.

If detecting a corner of someone's eye, in the final layer of the NN, we can have 2 more numbers in ouput called:

```
(lx, ly)
```

If detecting 4 corners of both eyes, it outputs 4 points:

```
(l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y)
```

If detecting 64 facial landmarks, you output 64 (x,y) coordinate.

So the output of the model is simply:

```
[l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y ... , l64x, l64y]
```

This is the output vector.

By selecting a number of landmarks and generating a label training sets that contains all of these landmarks, you can then have the neural network to tell you where are all the key positions or the key landmarks on a face.

```
InputImage -> ConvNet -> SetOfFeatures -> Output[face?, l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y ... , l64x, l64y]
```

In this example, output will have 129 units. (128 + 1 unit to say if there is face or not).

Application:
* Recognizing emotions from faces
* AR augmented reality filters (used in Snapchat to draw for example a crown on the face or other special effects.)
* Human pose detection, define a few key positions like the midpoint of the chest, the left shoulder, left elbow, the wrist, and so on, and just have a neural network to annotate key positions in the person's pose as well and by having a neural network output all of those points, we could also have the neural network output the pose of the person. And of course, to do that you also need to specify on these key landmarks like maybe l1x and l1y is the midpoint of the chest down to maybe l32x, l32y, if you use 32 coordinates to specify the pose of the person.

![alt text](_assets/Landmark.png)

The identity of landmark one must be consistent across different images.

## Object Detection
### Car detection example
Object detection is a task in computer vision where a model not only identifies what objects are in an image (like classification) but also locates them by drawing bounding boxes around each object. It's like classification plus localization for multiple objects.

In car detection for self-driving cars, the model scans an image of a road and draws red boxes around each car it finds, showing exactly where they are.

![alt text](_assets/CarDetectionExample.png)

Training set X will be many picture of cars closely cropped.

Train a ConvNet that inputs an image of car which is closely cropped and ouput is 0 or 1, whether it's a car or not a car.

Once the ConvNet is trained, we can use it in Sliding windows detection.

### Sliding windows detection
This is a basic method to detect objects by "sliding" a fixed-size window (like a square frame) across the entire image, cropping out each small section, and running it through a pre-trained ConvNet classifier to check if an object (e.g., a car) is in that crop. You repeat this process, moving the window step by step (called stride), until you've checked the whole image. To handle different object sizes, you repeat the process with larger or smaller windows.

* First, you train a ConvNet on tightly cropped images: positive examples are images that contain a car centered and filling the whole crop, negative examples are crops without cars (e.g., just road or trees).
* For a test image (like a road scene with two cars), you start with a small window (say 14x14 pixels) and slide it across the image with a stride of 2 pixels. For each position, crop the region, resize if needed, and feed it to the ConvNet.
* If the ConvNet says "car" for a crop, you mark that location as having a car.

![alt text](_assets/SlidingWindows1.png)

![alt text](_assets/SlidingWindows2.png)

* Then, repeat with a bigger window (e.g., 28x28) to catch larger cars, and so on for even bigger scales.

![alt text](_assets/SlidingWindows3.png)

![alt text](_assets/SlidingWindows4.png)

* In the video, Andrew shows a test image where sliding windows detect cars in overlapping regions, but notes this creates many duplicate detections that need cleaning up (though that's covered in later videos).

Stride is the number of pixels you move the window each time you slide it. A small stride (e.g., 1 or 2 pixels) means more overlaps and finer detection but takes longer; a large stride means faster but might miss objects.

In the car detection demo, using a stride of 2 pixels on a small window ensures the model checks overlapping areas, increasing the chance of catching a car that's not perfectly aligned with the window.

This refers to how much computer power and time the algorithm needs. The basic sliding windows method is inefficient because it runs the full ConvNet separately on every single crop, and for a large image with small strides and multiple scales, that could mean millions of computations.

Using coarse stride, a very big stride will reduce number of windows of we need to pass through the ConvNet, but it will hurt performance.

For a typical image, sliding a small window with a fine stride might require running the ConvNet thousands of times, making it too slow for real-time applications like self-driving cars. This is a big limitation, setting up for improvements in the next video (convolutional implementation to make it faster).

Train your classifier on high-quality, centered crops. Test on images with objects at various positions and sizes to see the algorithm's strengths and weaknesses.

## Convolutional implementation of sliding windows
Before deep learning, to detect objects (like cars, faces, cats), people would:

1. Take a small image patch (e.g., 64×64)
2. Slide it across the big image (left → right, top → bottom)
3. At each position, run a classifier to see if that patch contains the object.

This is called sliding windows.

The problem:

You must run a classifier thousands of times (for each window position), which is extremely slow.

### Turning FC layer into convolutional layers
If you take a CNN that ends with fully connected (FC) layers, you can convert the FC layers into convolutional layers.

Why?

Because fully connected layers are actually just 1×1 convolutions applied to the final feature map.

So the whole classifier can be transformed into a pure convolutional network.

Originally (image classifier):

```
Image  →  Conv  →  Conv  →  FC → Output (yes/no)
```

After converting the fully connected layer into convolution:

```
Big Image → Conv → Conv → Conv → (output grid)
```
Now the network produces a grid of predictions.

Each element in the grid corresponds to:

"Is there an object in this location?"

This grid is equivalent to the sliding windows, but computed all at once, not one window at a time.

You remove the cropping step. Feed the whole image into a ConvNet. The later layers naturally compute the classifier output for every possible window position because convolution = sliding a filter (like a window) over the image.

* Train a ConvNet on 14×14 cropped car images (same as before).
* Input: full road image (e.g., 100×100×3).
* The ConvNet has fully connected (FC) layers at the end → replace them with 1×1 convolutions so the whole network is fully convolutional.
* Output: a grid of predictions (e.g., 8×8 grid), where each cell says: “Is there a car centered in this 14×14 region?”
* Each cell in the output grid corresponds to one sliding window position!

![alt text](_assets/FCLayer1.png)

Change Y as four numbers, corresponding to the classes probabilities of the four classes that softmax units is classified amongst.

The full classes could be pedestrian, car, motorcycle, and background or something else. 

![alt text](_assets/FCLayer2.png)

What I'd like to do is show how these layers can be turned into convolutional layers. So, the ConvNet will draw same as before for the first few layers. 

One way of implementing this next layer, this fully connected layer is to implement this as a 5 by 5 filter and let's use 400 5 by 5 filters.

If you take a 5 by 5 by 16 image and convolve it with a 5 by 5 filter, remember, a 5 by 5 filter is implemented as 5 by 5 by 16 because our convention is that the filter looks across all 16 channels, so the outputs will be 1 by 1.

If you have 400 of these 5 by 5 by 16 filters, then the output dimension is going to be 1 by 1 by 400. So rather than viewing these 400 as just a set of nodes, we're going to view this as a 1 by 1 by 400 volume.

Mathematically, this is the same as a fully connected layer because each of these 400 nodes has a filter of dimension 5 by 5 by 16. So each of those 400 values is some arbitrary linear function of these 5 by 5 by 16 activations from the previous layer.

![alt text](_assets/ConvolutionalLayer1.png)

Next, to implement the next convolutional layer, we're going to implement a 1 by 1 convolution. If you have 400 1 by 1 filters then, with 400 filters the next layer will again be 1 by 1 by 400. So that gives you this next fully connected layer. 

![alt text](_assets/ConvolutionalLayer2.png)

Then finally, we're going to have another 1 by 1 filter, followed by a softmax activation. So as to give a 1 by 1 by 4 volume to take the place of these four numbers that the network was operating. 

![alt text](_assets/FCToConvolution.png)

This shows how you can take these fully connected layers and implement them using convolutional layers so that these sets of units instead are not implemented as 1 by 1 by 400 and 1 by 1 by 4 volumes.

After this conversion, let's see how you can have a convolutional implementation of sliding windows object detection. 

### Convolution implementation of sliding windows
ConvNet input: 14x14x3

![alt text](_assets/ConvNetInput.png)

Tested image: 16x16x3 (added yellow stripe to the border of this image)

In the original sliding windows algorithm, you might want to input the blue region into a convnet and run that once to generate a consecration 0 or 1 and then slightly down a bit, let say uses a stride of two pixels and then you might slide that to the right by two pixels to input this green rectangle into the convnet and we run the whole convnet and get another label, 0/1. Then you might input this orange region into the convnet and run it one more time to get another label. And then do it the fourth and final time with this lower right purple square. To run sliding windows on this 16 by 16 by 3 image is pretty small image. You run this ConvNet four times in order to get four labels. 

![alt text](_assets/TestedImage.png)

This is a lot of computations done by 4 ConvNets is highly duplicative.

What the convolutional implementation of sliding windows does is it allows these 4 classes in the ConvNet to share a lot of computation.

Specifically, here's what you can do. You can take the ConvNet and just run it same parameters, the same 5 by 5 filters, also 16 5 by 5 filters and run it. Now, you can have a 12 by 12 by 16 output volume. Then do the max pool, same as before. Now you have a 6 by 6 by 16, runs through your same 400, 5 by 5 filters to get now your 2 by 2 by 400 volume. 

So now instead of a 1 by 1 by 400 volume, we have instead a 2 by 2 by 400 volume. Run it through a 1 by 1 filter gives you another 2 by 2 by 400 instead of 1 by 1 like 400. Do that one more time and now you're left with a 2 by 2 by 4 output volume instead of 1 by 1 by 4.

It turns out that this blue 1 by 1 by 4 subset gives you the result of running in the upper left hand corner 14 by 14 image. This upper right 1 by 1 by 4 volume gives you the upper right result. The lower left gives you the results of implementing the convnet on the lower left 14 by 14 region. And the lower right 1 by 1 by 4 volume gives you the same result as running the convnet on the lower right 14 by 14 medium.

![alt text](_assets/Blue1x1x4.png)

![alt text](_assets/ConvSlidingWindows.png)

What this process does, what this convolution implementation does is, instead of forcing you to run four propagation on four subsets of the input image independently, Instead, it combines all four into one form of computation and shares a lot of the computation in the regions of image that are common. So all four of the 14 by 14 patches we saw here.

Old Way (Slow) = 4 Separate Runs

* Step 1: Cut out Square 1 → run brain → “Is there a car?”
* Step 2: Cut out Square 2 → run brain again
* Step 3: Cut out Square 3 → run brain again
* Step 4: Cut out Square 4 → run brain again

→ 4 full brain runs

→ Wastes time re-checking the same middle part 4 times!

New Way (Convolution) = 1 Run, Smart Sharing

* Step 1: Give whole photo to the brain once
* Step 2: The brain looks at everything together
* Step 3: It reuses the middle part for all 4 squares

→ 1 brain run gives answers for all 4 squares

→ No wasted work!

```text
Photo:
+-------------+
| 1     2     |
|   [CAR]     |
| 3     4     |
+-------------+

Overlap area = middle of the car
```

* Old way: checks middle 4 times
* New way: checks middle 1 time, shares it

![alt text](_assets/ConvSliding.png)

To recap, to implement sliding windows, previously, what you do is you crop out a region. Let's say this is 14 by 14 and run that through your convnet and do that for the next region over, then do that for the next 14 by 14 region, then the next one, then the next one, then the next one, then the next one and so on, until hopefully that one recognizes the car. 

But now, instead of doing it sequentially, with this convolutional implementation that you saw in the previous slide, you can implement the entire image, all maybe 28 by 28 and convolutionally make all the predictions at the same time by one forward pass through this big convnet and hopefully have it recognize the position of the car. 

![alt text](_assets/ConvSlidingWindows1.png)

Traditional sliding windows:
* Cut out thousands of patches
* Resize them
* Run the classifier individually many times

Convolutional implementation:
* Run the entire network one time
* Convolution naturally slides across the image
* Produces predictions at all locations in a single forward pass

So instead of computing:

```
Classifier(window_1)
Classifier(window_2)
Classifier(window_3)
...
```

You compute:

```
Classifier(big_image) → grid of results
```

Same idea, massively faster.

You trained a classifier on 64×64 images that ends with:
* Conv layers
* Then FC layers:

He shows you can convert the FC layer into:
* A 1×1 convolution layer

Then when you feed a bigger image, like 200×200, instead of 1 output, the network produces many outputs, such as a grid:

```
10 × 10 × 1
```

Each cell = classification for one sliding window.

But computed in one go.

Convolutions are translation-invariant.

If your classifier works for a 64×64 image, applying it on a big image via convolution:
* Automatically scans across the big image
* Tests every possible 64×64 region
* While sharing computation efficiently

It's like sliding windows but built into the CNN itself.

An FC layer is a layer where: Every input neuron is connected to every output neuron.

Just like in ordinary neural networks.

Example:

If the last convolutional layer outputs a tensor of shape:
```
7 × 7 × 512
```

We usually flatten it:

```
7*7*512 = 25088 values
```

Then feed it into an FC layer with, say, 4096 units.

So an FC layer is simply:

```
input:   [25088]
weights: [25088 × 4096]
output:  [4096]
```

This is just a big matrix multiplication.

Imagine you have a list of numbers:
```
[ a, b, c, d ]
```

And you want to combine them to get one output:
```
output = some weighted combination of a, b, c, d
```

That’s what an FC layer does.

So an FC layer is:
* you flatten everything into one long vector
* then multiply by weights
* and get outputs

It’s like:
```
[ a, b, c, d ] → FC Layer → [ output1, output2, output3, ... ]
```

Each output is just:
* multiply each input by some weight
* add them up
* add a bias

That’s it.

A 1×1 convolution: 
* Does NOT mix spatial locations
* ONLY mixes channels

If your feature map is:
```
7 × 7 × 512
```

Then a 1×1 filter with 4096 filters produces:
```
7 × 7 × 4096
```

At EACH pixel location, you look at all channels (colors) and combine them.

If your image (or feature map) has 3 channels (R, G, B):

A 1×1 convolution at that pixel is:
* take R, G, B
* multiply each by some weight
* add them up
* that gives 1 new value

But you can have multiple filters, so you can get:
* filter1 → output channel1
* filter2 → output channel2
* filter3 → output channel3
* …

So a 1×1 filter is basically an FC layer applied to a single pixel’s channels.

FC:

You flatten EVERYTHING into one long list.

1×1 Conv:

You DO NOT flatten.

You apply the “fully connected” math separately at each pixel location.

This is the key:

A 1×1 convolution is doing the SAME MATH as an FC layer, but doing it everywhere, not just once.

Let’s imagine you have a feature map:
```
7 x 7 x 512  (height x width x channels)
```

This means:
* 7 × 7 = 49 "pixels"
* each pixel has 512 numbers (channels)

How FC sees it

Flatten:
```
512, 512, 512, ... (49 times)
```

Then combine them into outputs with big matrix multiplication.

How 1×1 conv sees it

At each of the 49 positions:
* take the 512 channels
* combine them exactly like an FC layer would
* output however many channels you want

The math is EXACTLY the same as FC, but applied at each location.

* FC = combine all channels at ONE place
* 1×1 Conv = combine all channels at EVERY place

This is why they are equivalent.

An FC layer is just a 1×1 conv applied once.

A 1×1 conv is just an FC layer applied many times across the image.

In the old sliding window method:

You would crop tiny windows and run the FC-based classifier:
```
Window → CNN → FC → yes/no
```

now require repeating this MANY times.

But if you convert the FC into 1×1 conv:
```
Big Image → CNN → 1×1 conv
```

The network automatically scans for objects everywhere, because:

Convolution naturally slides across the image.

You get a GRID of outputs directly.

FC Layer

Operates on flattened vector once:
```
[##########] → FC → Result
```

1×1 Conv Layer

Operates on each spatial location:
```
[##][##][##]
[##][##][##] → 1×1 conv → grid of results
[##][##][##]
```

Same math, but done everywhere.

## Bounding Box Predictions
### Output accurate bounding boxes
Problem: The sliding window gives you a grid of squares (all 14×14). But cars are not always 14×14! They can be taller, wider, smaller, bigger.

Solution: Instead of fixed squares, predict the exact box around each car. The brain will output 4 numbers for every spot:

bx, by, bh, bw → where and how big the box is.

![alt text](_assets/BoundingBoxAccuracy.png)

"Turn classification into regression". "Don’t just say “car or no car”. Say: “Here’s the perfect red box around the car”

### YOLO algorithm
A good way to get this output more accurate bounding boxes is with the YOLO algorithm. YOLO stands for You Only Look Once, and is an algorithm due to Joseph Redman, Santosh Devala, Raj Gershwik, and Ali Fakhadi.

Here's what you do. Let's say you have an input image that 100 by 100. You're going to place down a grid on this image. And for the purposes of illustration, I'm going to use a 3 by 3 grid.

![alt text](_assets/100x100Image.png)

Although in an actual implementation, you'd use a finer one, like maybe a 19 by 19 grid.

Basic idea is you're going to take the image classification and localization algorithm that you saw in the first video of this week, and apply that to each of the nine grid cells of this image. 

For each of the nine grid cells, you specify a label Y, where the label Y is this eight-dimensional vector, same as you saw previously

![alt text](_assets/Label_y1.png)

In this image, we have 9 grid cells. So you'd have a vectorize this for each of the grid cells. 

Let's start to the upper left grid cell, this one up here. For that one, there is no object. So the label vector Y for the upper left grid cell would be 0, and then don't cares for the rest of these. The output label Y would be the same for all the grid cells with no interesting object in them.

![alt text](_assets/GridCellLabel_y.png)

This image has 2 objects, so YOLO algorithm takes the midpoint of each of the 2 objects, then assigns the object to the grid cell containing the midpoint.

![alt text](_assets/AssiningGridCell.png)

Even though the central grid cell has, you know, some parts of both cars will pretend the central grid cell has no interesting object. So for the central grid cell, the class label Y also looks like this vector with no object, and it's the first component PC, and then the rest are don't cares.

![alt text](_assets/GridCellLabel_y.png)

Whereas for cell that I've circled in green on the left, the target label Y would be as follows. There is an object, and then you write BX, BY, BH, BW to specify the position of this bounding box. If class 1 was a pedestrian, then that was 0, class 2 is a car, that's 1, class 3 was a motorcycle, that's 0.

And then similarly, for the grid cell on the right, because that does have an object in it, you know, it will also have some vector like this, as the target label corresponding to the grid cell on the right.

![alt text](_assets/GridCellLabel_y_1.png)

For each of these nine grid cells, you end up with a eight-dimensional output vector.

Because you have 3 by 3 grid cells, you have nine grid cells, the total volume of the output is going to be 3 by 3 by 8. So the target output is going to be 3 by 3 by 8, because you have 3 by 3 grid cells, and for each of the 3 by 3 grid cells, you have a eight-dimensional Y vector. So the target output volume is 3 by 3 by 8, where for example, this 1 by 1 by 8 volume in the upper left, corresponds to the target output vector for the upper left of the nine grid cells.

So for each of the 3 by 3 positions, for each of these nine grid cells, there's a corresponding eight-dimensional target vector Y that you want in the output, some of which could be don't cares if there's no object there.

To train your neural network, the input is a 100 by 100 by 3. Now that's the input image. And then you have a usual conf net with conf layers, max pool layers, and so on. So that in the end, you should choose the conf layers and the max pool layers and so on, so that this eventually maps to a 3 by 3 by 8 output volume.

What you do is you have an input X and you have target labels Y, which are 3 by 3 by 8, and you use backpropagation to train the neural network to map from any input X to this type of output volume Y.

![alt text](_assets/YOLO.png)

The advantage of this algorithm is that the neural network outputs precise bounding boxes. 

At test time, what you do is you feed in an input image X and run for a prop until you get this output Y. And then for each of the nine outputs, for each of the 3 by 3 positions in which the output, you can then just read off one or zero. Is there an object associated with that one of the nine positions?
* If there is an object, what object it is, and what is the bounding box for the objects in that grid cell?
* So long as you don't have more than one object in each grid cell, this algorithm should work okay.
* The problem of having multiple objects within a grid cell is something we'll address later. 

In practice, you might use a much finer grid, maybe 19 by 19. So you end up with 19 by 19 by 8. And that also makes your grid much finer and reduces the chance that there are multiple objects assigned to the same grid cell.

The way you assign an object to grid cell is you look at the midpoint of an object, and then you assign that object to whichever one grid cell contains the midpoint of the object. So each object, even if the object spans multiple grid cells, that object is assigned only to one of the nine grid cells or one of the 3 by 3 or one of the 19 by 19 grid cells. And with a 19 by 19 grid, the chance of a object of two midpoints of objects appearing in the same grid cell is just a bit smaller.

Notice two things. 
1. This is a lot like the image classification and localization algorithm, in that it outputs the bounding box coordinates explicitly. And so this allows the neural network to output bounding boxes of any aspect ratio, as well as output much more precise coordinates that aren't just dictated by the stride size of your sliding windows classifier.
2. This is a convolutional implementation. You're not implementing this algorithm nine times, on the 3 by 3 grid, or if you're using a 19 by 19 grid, 19 squared is 361. So you're not running the same algorithm, you know, 361 times or 19 squared times. Instead, this is one single convolutional implementation where you use one conf net with a lot of shared computation between all the computations needed for all of your, you know, 3 by 3 or all of your 19 by 19 grid cells. So this is a pretty efficient algorithm. 

One nice thing about the YOLO algorithm, which accounts for its popularity, is because this is a convolutional implementation, it actually runs very fast. So this works even for real-time object detection.

### Specify the bounding boxes
Given these two cars, remember we have the 3 by 3 grid. Let's take the example of the car on the right. 

In this grid cell, there is an object and so the target label Y will be

![alt text](_assets/SpecifyBoundingBoxes1.png)

In the YOLO algorithm, relative to this square, we're going to take the convention that the upper left point here is (0, 0), and this lower right point is (1, 1).

![alt text](_assets/Relative.png)

To specify the position of that midpoint, that orange dot, BX might be- let's see, X looks like it's about 0.4, since it's maybe about 0.4 of the way to the right, and then Y looks like that's maybe 0.3.

Then the height of the bounding box is specified as a fraction of the overall width of this box. So the width of this red box is maybe 90 percent of that blue line. And so bh, 0.5. And the height of this is maybe one-half of the overall height of the grid cell. So bw would be, let's say, 0.9.

In other words, this BX, BY, BH, BW are specified relative to the grid cell. 

bx and by, this has to be between 0 and 1. Because pretty much by definition, that orange dot is within the bounds of that grid cell it's assigned to. If it wasn't between 0 and 1, if it was outside the square, then it would have been assigned to a different grid cell.

But bh and bw could be greater than 1. In particular, if you had a car where the bounding box was that, then the height and width of a bounding box, this could be greater than 1. 

![alt text](_assets/BoundingBoxesRange.png)






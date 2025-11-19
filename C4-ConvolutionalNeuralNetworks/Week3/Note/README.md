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

## Intersection Over Union
Intersection Over Union is used for:
* Evaluating object detection algorithm
* Adding another component to object detection algorithm to make it work better.

### Evaluating object localization
In objection detection task, we expect to localize the object.

![alt text](_assets/LocalizationPosition.png)

If the red box the ground-truth bounding box, and if your algorithm outputs this bounding box in purple, is this a good outcome or a bad one?

What the intersection over union function does, or IoU does, is it computes the intersection over union of these two bounding boxes. 

The union of the 2 bounding boxes is the area that is contained in either bounding boxes.

![alt text](_assets/Union2Boxes.png)

Intersection of 2 bounding boxes

![alt text](_assets/Intersection2Boxes.png)

Intersection Over Union = ${sizeOfIntersection \over sizeOfUnion}$

![alt text](_assets/IoU.png)

By convension, the low compute devision task will judge that your answer is correct if the IoU is greater than 0.5.

If the predicted and the ground-truth bounding boxes overlapped perfectly, the IoU would be 1, because the intersection would equal to the union. 

In general, so long as the IoU is greater than or equal to 0.5, then the answer will look okay, look pretty decent.

By convention, very often 0.5 is used as a threshold to judge as whether the predicted bounding box is correct or not. 

If you want to be more stringent, you can judge an answer as correct, only if the IoU is greater than equal to 0.6 or some other number. The higher the IoUs, the more accurate the bounding the box.

This is one way to map localization, to accuracy where you just count up the number of times an algorithm correctly detects and localizes an object where you could use a definition like this, of whether or not the object is correctly localized.

0.5 is just a human chosen convention. There's no particularly deep theoretical reason for it. You can also choose some other threshold like 0.6 if you want to be more stringent. 

More generally, IoU is a measure of the overlap between two bounding boxes.

Where if you have two boxes, you can compute the intersection, compute the union, and take the ratio of the two areas. This is also a way of measuring how similar two boxes are to each other. 

## Non-max Suppression
One of the problems of Object Detection is that the algorithm may find multiple detections of the same objects. Non-max Suppression is a way to make sure that the algorithm detects each object only once.

### Non-max suppression example
Let's say you want to detect pedestrians, cars, and motorcycles in this image.

![alt text](_assets/Non-maxExample.png)

You might place a grid over this, and this is a 19 by 19 grid. 

![alt text](_assets/Non-maxExampleGrid.png)

While technically the car on the right has just one midpoint, so it should be assigned just one grid cell. The car on the left also has 1 midpoint, so only 1 of those grid cells should predict there is a car.

In practice, you're running an object classification and localization algorithm for every one of these split cells. So it's quite possible that many split cells might think that the center of a car is in it.

Because you're running the image classification and localization algorithm on every grid cell, on 361 grid cells, it's possible that many of them will raise their hand and say, "My Pc, my chance of thinking I have an object in it is large." Rather than just having two of the grid cells out of the 19 squared or 361 think they have detected an object. 

-> You end up with multiple detections of each object.

![alt text](_assets/MultipleDetections.png)

What non-max suppression does, is it cleans up these detections. So they end up with just one detection per car, rather than multiple detections per car.

Concretely, what it does, is it first looks at the probabilities associated with each of these detections which is Pc (the probability of a detection). And it first takes the largest one, which in this case is 0.9 and says, "That's my most confident detection, so let's highlight that and just say I found the car there." 

![alt text](_assets/MultipleDetections.png)

Having done that the non-max suppression part then looks at all of the remaining rectangles and all the ones with a high overlap, with a high IOU, with this one that you've just output will get suppressed. So those two rectangles with the 0.6 and the 0.7. Both of those overlap a lot with the light blue rectangle. So those, you are going to suppress and darken them to show that they are being suppressed.

Next, you then go through the remaining rectangles and find the one with the highest probability, the highest Pc, which in this case is this one with 0.8. So let's commit to that and just say, "Oh, I've detected a car there." And then, the non-max suppression part is to then get rid of any other ones with a high IOU. 

![alt text](_assets/HighlightedNonMax.png)

Non-max means that you're going to output your maximal probabilities classifications but suppress the close-by ones that are non-maximal.

Let's go through the details of the algorithm. First, on this 19 by 19 grid, you're going to get a 19 by 19 by 8 output volume. 

![alt text](_assets/19x19x8Example.png)

Although, for this example, I'm going to simplify it to say that you only doing car detection. So, let me get rid of the C1, C2, C3, and pretend for this line, that each output for each of the 19 by 19, so for each of the 361, which is 19 squared, for each of the 361 positions, you get an output prediction of the following. 

![alt text](_assets/PredictionExample.png)

Which is the chance there's an object, and then the bounding box. And if you have only one object, there's no C1, C2, C3 prediction. The details of what happens, you have multiple objects, I'll leave to the programming exercise, which you'll work on towards the end of this week. 

To intimate non-max suppression, the first thing you can do is discard all the boxes, discard all the predictions of the bounding boxes with Pc less than or equal to some threshold, let's say 0.6. 

We're going to say that unless you think there's at least a 0.6 chance it is an object there, let's just get rid of it. This has caused all of the low probability output boxes. 

The way to think about this is for each of the 361 positions, you output a bounding box together with a probability of that bounding box being a good one. So we're just going to discard all the bounding boxes that were assigned a low probability.

Next, while there are any remaining bounding boxes that you've not yet discarded or processed, you're going to
* Repeatedly pick the box with the highest probability, with the highest Pc, and then output that as a prediction. So this is a process on a previous slide of taking one of the bounding boxes, and making it lighter in color. So you commit to outputting that as a prediction for that there is a car there.
* Next, you then discard any remaining box. Any box that you have not output as a prediction, and that was not previously discarded. So discard any remaining box with a high overlap, with a high IOU, with the box that you just output in the previous step. This second step in the while loop was when on the previous slide you would darken any remaining bounding box that had a high overlap with the bounding box that we just made lighter, that we just highlighted.

You keep doing this while there's still any remaining boxes that you've not yet processed, until you've taken each of the boxes and either output it as a prediction, or discarded it as having too high an overlap, or too high an IOU, with one of the boxes that you have just output as your predicted position for one of the detected objects.


I've described the algorithm using just a single object on this slide. If you actually tried to detect three objects say pedestrians, cars, and motorcycles, then the output vector will have three additional components. And it turns out, the right thing to do is to independently carry out non-max suppression three times, one on each of the outputs classes. 

## Anchor Boxes
One of the problems with object detection is that each of the grid cells can detect only one object. What if a grid cell wants to detect multiple objects? You can use the idea of anchor boxes. 

### Overlapping objects
Let's start with an example. Let's say you have an image like this. And for this example, I am going to continue to use a 3 by 3 grid. 

![alt text](_assets/Overlapping.png)

Notice that the midpoint of the pedestrian and the midpoint of the car are in almost the same place and both of them fall into the same grid cell.

For that grid cell, if Y outputs this vector where you are detecting three classes: pedestrians, cars and motorcycles, it won't be able to output two detections. So I have to pick one of the two detections to output.

![alt text](_assets/Y-vector.png)

With the idea of anchor boxes, what you are going to do is pre-define two different shapes called, anchor boxes or anchor box shapes.

![alt text](_assets/AnchorBoxes.png)

Then associate two predictions with the two anchor boxes.

In general, you might use more anchor boxes, maybe five or even more. 

So vector Y will be repeated twice in the same vector:

![alt text](_assets/RepeatedY.png)

Because the shape of the pedestrian is more similar to the shape of anchor box 1 and anchor box 2, you can use the first 8 numbers to encode that PC as 1, yes there is a pedestrian.

Use the first bx, by, bh, bw to encode the bounding box around the pedestrian, and then use first c1, c2, c3 to encode that that object is a pedestrian.

Then because the box around the car is more similar to the shape of anchor box 2 than anchor box 1, you can then use the second 8 numbers to encode that the second object is the car, and have the bounding box and so on be all the parameters associated with the detected car.

To summarize, previously, before you are using anchor boxes, for each object in the training set and the training set image, it was assigned to the grid cell that corresponds to that object's midpoint. Output Y was 3x3x8, because we have 3x3 grid, and for each grid position, we had output vector Y.

With 2 anchor boxes, each object is assigned to the same grid cell as before, assigned to the grid cell that contains the object's midpoint, but it is assigned to a grid cell and anchor box with the highest IoU with the object's shape. So, you have two anchor boxes, you will take an object and see which of the two anchor boxes has a higher IoU, will be drawn through bounding box. Whichever it is, that object then gets assigned not just to a grid cell but to a pair. It gets assigned to grid cell comma anchor box pair. And that's how that object gets encoded in the target label. 

The output Y is going to be 3 by 3 by 16. Because Y is now 16 dimensional. Or if you want, you can also view this as 3 by 3 by 2 by 8 ,because there are now two anchor boxes and Y is eight dimensional. And dimension of Y being 8 was because we have 3 objects classes; if you have more objects than the dimension of Y would be even higher. 

### Anchor box example

![alt text](_assets/AnchorBoxExample.png)

The pedestrian is more similar to the shape of anchor box 1. So for the pedestrian, we're going to assign it to the top half of this vector. So yes, there is an object, there will be some bounding box associated at the pedestrian. And I guess if a pedestrian is class 1, then c1 is 1, and then zero, zero. And then the shape of the car is more similar to anchor box 2. And so the rest of this vector will be 1 and then the bounding box associated with the car, and then the car is C2, so there's 0, 1, 0. And so that's the label Y for that lower middle grid cell that this arrow was pointing to. 

![alt text](_assets/AnchorBoxesOutput.png)

Now, what if this grid cell only had a car and had no pedestrian? 

If it only had a car, then assuming that the shape of the bounding box around the car is still more similar to anchor box 2, then the target label Y, if there was just a car there and the pedestrian had gone away, it will still be the same for the anchor box 2 component. Remember that this is a part of the vector corresponding to anchor box 2. And for the part of the vector corresponding to anchor box 1, what you do is you just say there is no object there. So PC is zero, and then the rest of these will be don't cares. 

![alt text](_assets/CarOnly.png)

What if you have two anchor boxes but three objects in the same grid cell? That's one case that this algorithm doesn't handle well. Hopefully, it won't happen. But if it does, this algorithm doesn't have a great way of handling it. I will just influence some default tiebreaker for that case.

What if you have two objects associated with the same grid cell, but both of them have the same anchor box shape? Again, that's another case that this algorithm doesn't handle well. If you influence some default way of tiebreaking if that happens, hopefully this won't happen with your data set, it won't happen much at all. And so, it shouldn't affect performance as much.

Even though I'd motivated anchor boxes as a way to deal with what happens if two objects appear in the same grid cell, in practice, that happens quite rarely, especially if you use a 19 by 19 rather than a 3 by 3 grid. The chance of two objects having the same midpoint rather these 361 cells, it does happen, but it doesn't happen that often. Maybe even better motivation or even better results that anchor boxes gives you is it allows your learning algorithm to specialize better. 

In particular, if your data set has some tall, skinny objects like pedestrians, and some wide objects like cars, then this allows your learning algorithm to specialize so that some of the outputs can specialize in detecting wide, fat objects like cars, and some of the output units can specialize in detecting tall, skinny objects like pedestrians.

Finally, how do you choose the anchor boxes?

People used to just choose them by hand or choose maybe 5 or 10 anchor box shapes that spans a variety of shapes that seems to cover the types of objects you seem to detect. As a much more advanced version, just in the advance common for those of who have other knowledge in machine learning, and even better way to do this in one of the later YOLO research papers, is to use a K-means algorithm, to group together two types of objects shapes you tend to get and then to use that to select a set of anchor boxes that this most stereotypically representative of the maybe multiple, of the maybe dozens of object classes you're trying to detect. But that's a more advanced way to automatically choose the anchor boxes. If you just choose by hand a variety of shapes that reasonably expands the set of object shapes, you expect to detect some tall, skinny ones, some fat, white ones. That should work with these as well.






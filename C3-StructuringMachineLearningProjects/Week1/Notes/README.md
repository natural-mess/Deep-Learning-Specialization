# Week 1: ML Strategy

**Learning Objectives**
* Explain why Machine Learning strategy is important
* Apply satisficing and optimizing metrics to set up your goal for ML projects
* Choose a correct train/dev/test split of your dataset
* Define human-level performance
* Use human-level performance to define key priorities in ML projects
* Take the correct ML Strategic decision based on observations of performances and dataset

## Why ML Strategy
Let's say you are working on your cat classifier. And after working it for some time, you've gotten your system to have 90% accuracy, but this isn't good enough for your application. You might then have a lot of ideas as to how to improve your system. 

Ideas:
* Collect more data
* Collect more diverse training set
* Train algorithm longer with gradient descent
* Try Adam instead of gradient descent
* Try bigger network
* Try smaller network
* Try dropout
* Add $L_2$ regularization
* Network architecture
  * Activation functions
  * Change number of hidden units
  * …

![alt text](_assets/MotivatingExample.png)

When trying to improve a deep learning system, you often have a lot of ideas or things you could try. 

And the problem is that if you choose poorly, it is entirely possible that you end up spending six months charging in some direction only to realize after six months that that didn't do any good.

For example, I've seen some teams spend literally six months collecting more data only to realize after six months that it barely improved the performance of their system.

Assuming you don't have six months to waste on your problem, won't it be nice if you had quick and effective ways to figure out which of all of these ideas and maybe even other ideas, are worth pursuing and which ones you can safely discard. 

This course teaches you
* A number of strategies, that is, ways of analyzing a machine learning problem that will point you in the direction of the most promising things to try. 
* Share with you a number of lessons I've learned through building and shipping large number of deep learning products.

## Orthogonalization
### TV tuning example
One of the challenges with building machine learning systems is that there's so many things you could try, so many things you could change. Including, for example, so many hyperparameters you could tune.

Most effective machine learning people are very clear-eyed about what to tune in order to try to achieve one effect. This is a process we call orthogonalization. 

Here's a picture of an old school television, with a lot of knobs that you could tune to adjust the picture in various ways. 

![alt text](_assets/TVTuning.png)

So for these old TV sets, maybe there was one knob to adjust how tall vertically your image is and another knob to adjust how wide it is. Maybe another knob to adjust how trapezoidal it is, another knob to adjust how much to move the picture left and right, another one to adjust how much the picture's rotated, and so on.

What TV designers had spent a lot of time doing was to build the circuitry, really often analog circuitry back then, to make sure each of the knobs had a relatively interpretable function. 

In contrast, imagine if you had a knob that tunes 0.1 x how tall the image is, + 0.3 x how wide the image is,- 1.7 x how trapezoidal the image is, + 0.8 times the position of the image on the horizontal axis, and so on.

![alt text](_assets/KnobTuning.png)

If you tune this knob, then the height of the image, the width of the image, how trapezoidal it is, how much it shifts, it all changes all at the same time.

If you have a knob like that, it'd be almost impossible to tune the TV so that the picture gets centered in the display area. 

In this context, orthogonalization refers to that the TV designers had designed the knobs so that each knob kind of does only one thing. And this makes it much easier to tune the TV, so that the picture gets centered where you want it to be.

Here's another example of orthogonalization. If you think about learning to drive a car, a car has three main controls, which are:
* Steering, the steering wheel decides how much you go left or right. 
* Acceleration
* Braking

So these three controls, or really one control for steering and another two controls for your speed, it makes it relatively interpretable, what your different actions through different controls will do to your car. 

But now imagine if someone were to build a car so that there was a joystick, where:
* One axis of the joystick controls 0.3 x your steering angle - 0.8 x your speed. 
* A different control that controls 2 x the steering angle + 0.9 x the speed of your car. 

In theory, by tuning these two knobs, you could get your car to steer at the angle and at the speed you want. But it's much harder than if you had just one single control for controlling the steering angle, and a separate, distinct set of controls for controlling the speed.

So the concept of orthogonalization refers to that, if you think of one dimension of what you want to do as controlling a steering angle, and another dimension as controlling your speed. Then you want one knob to just affect the steering angle as much as possible, and another knob, in the case of the car, is really acceleration and braking, that controls your speed.

But if you had a control that mixes the two together, like a control like this one that affects both your steering angle and your speed, something that changes both at the same time, then it becomes much harder to set the car to the speed and angle you want.

And by having orthogonal, orthogonal means at 90 degrees to each other. By having orthogonal controls that are ideally aligned with the things you actually want to control, it makes it much easier to tune the knobs you have to tune. To tune the steering wheel angle, and your accelerator, your braking, to get the car to do what you want.

![alt text](_assets/CarTuning.png)

### Chain of assumptions in ML
How does this relate to machine learning?

For a supervised learning system to do well, you usually need to tune the knobs of your system to make sure that four things hold true. 
* First, is that you usually have to make sure that you're at least doing well on the training set. So performance on the training set needs to pass some acceptability assessment. For some applications, this might mean doing comparably to human level performance. But this will depend on your application.
* After doing well on the training sets, you then hope that this leads to also doing well on the dev set. 
* And you then hope that this also does well on the test set on cost function on the cost function. 
* And finally, you hope that doing well on the test set on the cost function results in your system performing in the real world. So you hope that this resolves in happy cat picture app users, for example. 

To relate back to the TV tuning example, if the picture of your TV was either too wide or too narrow, you wanted one knob to tune in order to adjust that. You don't want to have to carefully adjust five different knobs, which also affect different things. You want one knob to just affect the width of your TV image.

In a similar way, if your algorithm is not fitting the training set well on the cost function, you want one knob, or maybe one specific set of knobs that you can use, to make sure you can tune your algorithm to make it fit well on the training set. 

The knobs you use to tune this are:
* You might train a bigger network. 
* Or you might switch to a better optimization algorithm, like the Adam optimization algorithm, and so on
* Some other options we'll discuss later this week and next week.

In contrast, if you find that the algorithm is not fitting the dev set well, then there's a separate set of knobs. Yes, that's my not very artistic rendering of another knob, you want to have a distinct set of knobs to try. So for example, if your algorithm is not doing well on the dev set, it's doing well on the training set but not on the dev set, then you have a set of knobs:
* Regularization that you can use to try to make it satisfy the second criteria. So the analogy is, now that you've tuned the width of your TV set, if the height of the image isn't quite right, then you want a different knob in order to tune the height of the TV image. And you want to do this hopefully without affecting the width of your TV image too much.
* Getting a bigger training set would be another knob you could use, that helps your learning algorithm generalize better to the dev set.

Now, having adjusted the width and height of your TV image, well, what if it doesn't meet the third criteria? What if you do well on the dev set but not on the test set?

If that happens, then the knob you tune is:
* You probably want to get a bigger dev set. Because if it does well on the dev set but not the test set, it probably means you've overtuned to your dev set, and you need to go back and find a bigger dev set. 

And finally, if it does well on the test set, but it isn't delivering to you a happy cat picture app user, then what that means is that you want to go back and change either the dev set or the cost function. Because if doing well on the test set according to some cost function doesn't correspond to your algorithm doing what you need it to do in the real world, then it means that either your dev/test set distribution isn't set correctly, or your cost function isn't measuring the right thing.

![alt text](_assets/ChainOfAssumption.png)

When I train a neural network, I tend not to use early stopping. It's not a bad technique, quite a lot of people do it. But I personally find early stopping difficult to think about. Because this is a knob that simultaneously affects: 
* How well you fit the training set, because if you stop early, you fit the training set less well.
* It also simultaneously is often done to improve your dev set performance.

So this is one knob that is less orthogonalized, because it simultaneously affects two things. It's like a knob that simultaneously affects both the width and the height of your TV image. And it doesn't mean that it's a bad knob to use, you can use it if you want. But when you have more orthogonalized controls, such as these other ones that I'm writing down here, then it just makes the process of tuning your network much easier.

## Single Number Evaluation Metric
Whether you're tuning hyperparameters, or trying out different ideas for learning algorithms, or just trying out different options for building your machine learning system. You'll find that your progress will be much faster if you have a single real number evaluation metric that lets you quickly tell if the new thing you just tried is working better or worse than your last idea. So when teams are starting on a machine learning project, I often recommend that you set up a single real number evaluation metric for your problem. 

### Using a single number evaluation metric
You've heard me say before that applied machine learning is a very empirical process. We often have an idea, code it up, run the experiment to see how it did, and then use the outcome of the experiment to refine your ideas. And then keep going around this loop as you keep on improving your algorithm. 

![alt text](_assets/MLProcess.png)

Let's say for your cat classifier, you had previously built some classifier A. And by changing the hyperparameters and the training sets or some other thing, you've now trained a new classifier, B.

![alt text](_assets/ClassifierA-B.png)

One reasonable way to evaluate the performance of your classifiers is to look at its precision and recall. The exact details of what's precision and recall don't matter too much for this example. 

The definition of **precision** is, of the examples that your classifier recognizes as cats, what percentage actually are cats? So if classifier A has 95% precision, this means that when classifier A says something is a cat, there's a 95% chance it really is a cat. 

**Recall** is, of all the images that really are cats, what percentage were correctly recognized by your classifier? So what percentage of actual cats are correctly recognized? So if classifier A is 90% recall, this means that of all of the images in, say, your dev sets that really are cats, classifier A accurately pulled out 90% of them.

It turns out that there's often a tradeoff between precision and recall, and you care about both. You want that, when the classifier says something is a cat, there's a high chance it really is a cat. But of all the images that are cats, you also want it to pull a large fraction of them as cats. So it might be reasonable to try to evaluate the classifiers in terms of its precision and its recall.

The problem with using precision recall as your evaluation metric is that if classifier A does better on recall, which it does here, the classifier B does better on precision, then you're not sure which classifier is better.

If you're trying out a lot of different ideas, a lot of different hyperparameters, you want to rather quickly try out not just two classifiers, but maybe a dozen classifiers and quickly pick out the "best ones", so you can keep on iterating from there. 

With two evaluation metrics, it is difficult to know how to quickly pick one of the two or quickly pick one of the ten. So what I recommend is rather than using two numbers, precision and recall, to pick a classifier, you just have to find a new evaluation metric that combines precision and recall.

In the machine learning literature, the standard way to combine precision and recall is something called an F1 score. And the details of F1 score aren't too important, but informally, you can think of this as the average of precision, P, and recall, R. Formally, the F1 score is defined by this formula

$({2 \over {1 \over {{1 \over P} + {1 \over R}}}})$

In mathematics, this function is called the "harmonic mean" of precision P and recall R.

Less formally, you can think of this as some way that averages precision and recall. Only instead of taking the arithmetic mean, you take the harmonic mean, which is defined by this formula. And it has some advantages in terms of trading off precision and recall. 

![alt text](_assets/F1Score.png)

In this example, you can then see right away that classifier A has a better F1 score. And assuming F1 score is a reasonable way to combine precision and recall, you can then quickly select classifier A over classifier B.

What I found for a lot of machine learning teams is that having a well-defined dev set, which is how you're measuring precision and recall, plus a single number evaluation metric, sometimes I'll call it single row number evaluation metric allows you to quickly tell if classifier A or classifier B is better, and therefore having a dev set plus single number evaluation metric distance to speed up iterating. It speeds up this iterative process of improving your machine learning algorithm.

### Another example
Let's look at another example. Let's say you're building a cat app for cat lovers in four major geographies, the US, China, India, and other, the rest of the world. And let's say that your two classifiers achieve different errors in data from these four different geographies. Algorithm A achieves 3% error on pictures submitted by US users and so on.

![alt text](_assets/AnotherExample.png)

It might be reasonable to keep track of how well your classifiers do in these different markets or these different geographies. But by tracking four numbers, it's very difficult to look at these numbers and quickly decide if algorithm A or algorithm B is superior. And if you're testing a lot of different classifiers, then it's just difficult to look at all these numbers and quickly pick one. 

![alt text](_assets/AnotherExample2.png)

What I recommend in this example is, in addition to tracking your performance in the four different geographies, to also compute the average.

![alt text](_assets/AnotherExampleAverage.png)

And assuming that average performance is a reasonable single real number evaluation metric, by computing the average, you can quickly tell that it looks like algorithm C has a lowest average error. And you might then go ahead with that one. If you have to pick an algorithm to keep on iterating from.

So your work load machine learning is often, you have an idea, you implement it and try it out, and you want to know whether your idea helped. So what we've seen in this video is that having a single number evaluation metric can really improve your efficiency or the efficiency of your team in making those decisions.

## Satisficing and Optimizing Metric
It's not always easy to combine all the things you care about into a single row number evaluation metric. In those cases I've found it sometimes useful to set up satisficing as well as optimizing metrics.

### Another cat classification example
Let's say that you've decided you care about the classification accuracy of your cat's classifier, this could have been F1 score or some other measure of accuracy

![alt text](_assets/CatClassifier.png)

But let's say that in addition to accuracy you also care about the running time. So how long it takes to classify an image and classifier A takes 80 milliseconds, B takes 95 milliseconds, and C takes 1,500 milliseconds, that's 1.5 seconds to classify an image. 

One thing you could do is combine accuracy and running time into an overall evaluation metric. And so the costs such as maybe the overall cost is accuracy minus 0.5 times running time. 

$cost = accuracy - 0.5*runningTime$

Maybe it seems a bit artificial to combine accuracy and running time using a formula like this, like a linear weighted sum of these two things.

Here's something else you could do instead which is that you might want to choose a classifier that maximizes accuracy but subject to that the running time, that is the time it takes to classify an image, that that has to be less than or equal to 100 milliseconds. 

In this case we would say that accuracy is an optimizing metric because you want to maximize accuracy. You want to do as well as possible on accuracy but that running time is what we call a satisficing metric. Meaning that it just has to be good enough, it just needs to be less than 100 milliseconds and beyond that you don't really care, or at least you don't care that much.

So this will be a pretty reasonable way to trade off or to put together accuracy as well as running time. And it may be the case that so long as the running time is less that 100 milliseconds, your users won't care that much whether it's 100 milliseconds or 50 milliseconds or even faster. 

And by defining optimizing as well as satisficing metrics, this gives you a clear way to pick the "best classifier", which in this case would be classifier B because of all the ones with a running time better than 100 milliseconds, it has the best accuracy.

![alt text](_assets/CatExample.png)

More generally
* If you have N metrics that you care about it's sometimes reasonable to pick one of them to be optimizing. So you want to do as well as is possible on that one.
* Then N minus 1 to be satisficing, meaning that so long as they reach some threshold such as running times faster than 100 milliseconds, but so long as they reach some threshold, you don't care how much better it is in that threshold, but they have to reach that threshold.

Here's another example. Let's say you're building a system to detect wake words, also called trigger words. So this refers to the voice control devices like the Amazon Echo where you wake up by saying Alexa or some Google devices which you wake up by saying okay Google or some Apple devices which you wake up by saying Hey Siri or some Baidu devices which you wake up by saying you ni hao Baidu.

![alt text](_assets/WakeWordExample.png)

These are the wake words you use to tell one of these voice control devices to wake up and listen to something you want to say.

So you might care about the accuracy of your trigger word detection system. So when someone says one of these trigger words, 
* How likely are you to actually wake up your device
* You might also care about the number of false positives. When no one actually said this trigger word, how often does it randomly wake up?

So in this case maybe one reasonable way of combining these two evaluation metrics might be to 
* Maximize accuracy, so when someone says one of the trigger words, maximize the chance that your device wakes up. 
* subject to that, you have at most one false positive every 24 hours of operation. So that your device randomly wakes up only once per day on average when no one is actually talking to it. 

So in this case
* Accuracy is the optimizing metric
* A number of false positives every 24 hours is the satisficing metric where you'd be satisfied so long as there is at most one false positive every 24 hours. 

![alt text](_assets/MetricExamples.png)

To summarize, if there are multiple things you care about by say there's one as the optimizing metric that you want to do as well as possible on and one or more as satisficing metrics were you'll be satisfice. So long as it does better than some threshold you can now have an almost automatic way of quickly looking at multiple core size and picking the, quote, "best one".

Now these evaluation metrics must be evaluated or calculated on a training set or a development set or maybe on the test set. So one of the things you also need to do is set up training, dev or development, as well as test sets. 

## Train/Dev/Test Distributions
The way you set up your training dev, or development sets and test sets, can have a huge impact on how rapidly you or your team can make progress on building machine learning application. The same teams, even teams in very large companies, set up these data sets in ways that really slows down, rather than speeds up, the progress of the team. 
### Cat classification dev/test sets
In this video, I want to focus on how you set up your dev and test sets.

The dev set is also called the development set, or sometimes called the hold out cross validation set.

Workflow in machine learning is that you try a lot of ideas, train up different models on the training set, and then use the dev set to evaluate the different ideas and pick one. And, keep innovating to improve dev set performance until, finally, you have one clause that you're happy with that you then evaluate on your test set.

Now, let's say, by way of example, that you're building a cat crossfire, and you are operating in these regions: 

* US
* UK
* Other Europe
* South America
* India
* China
* Other Asia
* Australia

So, how do you set up your dev set and your test set? 

One way you could do so is to pick four of these regions. I'm going to use these four but it could be four randomly chosen regions. And say, that data from these four regions will go into the dev set. And, the other four regions, I'm going to use these four, could be randomly chosen four as well, that those will go into the test set.

![alt text](_assets/DevTestSet.png)

It turns out, this is a very bad idea because in this example, your dev and test sets come from different distributions. 

I would, instead, recommend that you find a way to make your dev and test sets come from the same distribution. 

One picture to keep in mind is that, I think, setting up your dev set, plus, your single role number evaluation metric, that's like placing a target and telling your team where you think is the bull's eye you want to aim at. Because, what happen once you've established that dev set and the metric is that, the team can innovate very quickly, try different ideas, run experiments and very quickly use the dev set and the metric to evaluate crossfires and try to pick the best one. 

Machine learning teams are often very good at shooting different arrows into targets and innovating to get closer and closer to hitting the bullseye. So, doing well on your metric on your dev sets.

The problem with how we've set up the dev and test sets in the example on the left is that, your team might spend months innovating to do well on the dev set only to realize that, when you finally go to test them on the test set, that data from these four countries or these four regions at the bottom, might be very different than the regions in your dev set. So, you might have a nasty surprise and realize that, all the months of work you spent optimizing to the dev set, is not giving you good performance on the test set.

So, having dev and test sets from different distributions is like setting a target, having your team spend months trying to aim closer and closer to bull's eye, only to realize after months of work that, you'll say, "Oh wait, to test it, I'm going to move target over here." 

![alt text](_assets/Target.png)

And, the team might say, "Well, why did you make us spend months optimizing for a different bull's eye when suddenly, you can move the bull's eye to a different location somewhere else?" 

To avoid this, what I recommend instead is that, you take all this randomly shuffled data into the dev and test set. So that, both the dev and test sets have data from all eight regions and that the dev and test sets really come from the same distribution, which is the distribution of all of your data mixed together. 

![alt text](_assets/DevTestSet2.png)

### True story (details changed)
Here's another example. This is a, actually, true story but with some details changed. 

I know a machine learning team that actually spent several months optimizing on a dev set which was comprised of loan approvals for medium income zip codes.

So, the specific machine learning problem was, "Given an input X about a loan application, can you predict y and which is, whether or not, they'll repay the loan?" So, this helps you decide whether or not to approve a loan. 

The dev set came from loan applications. They came from medium income zip codes. Zip codes is what we call postal codes in the United States. 

After working on this for a few months, the team then, suddenly decided to test this on data from low income zip codes or low income postal codes. 

And, of course, the distributional data for medium income and low income zip codes is very different. And, the crossfire, that they spend so much time optimizing in the former case, just didn't work well at all on the latter case. And so, this particular team actually wasted about three months of time and had to go back and really re-do a lot of work.

What happened here was, the team spent three months aiming for one target, and then, after three months, the manager asked, "Oh, how are you doing on hitting this other target?" This is a totally different location. And, it just was a very frustrating experience for the team.

![alt text](_assets/DetailsChanged.png)

### Guideline
So, what I recommand for setting up a dev set and test set is, **choose a dev set and test set to reflect data you expect to get in future and consider important to do well on.** 

**In particular, the dev set and the test set here, should come from the same distribution.** 

So, whatever type of data you expect to get in the future, and want to do well on, try to get data that looks like that. And, whatever that data is, put it into both your dev set and your test set. Because that way, you're putting the target where you actually want to hit and you're having the team innovate very efficiently to hitting that same target, hopefully, the same target well. 

The important take away from this video is that, setting up the dev set, as well as the valuation metric, is really defining what target you want to aim at. And hopefully, by setting the dev set and the test set to the same distribution, you're really aiming at whatever target you hope your machine learning team will hit. 

The way you choose your training set will affect how well you can actually hit that target. 

## Size of the Dev and Test Sets




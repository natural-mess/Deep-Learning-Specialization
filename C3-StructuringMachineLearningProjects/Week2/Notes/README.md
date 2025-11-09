# Week 2: ML Strategy

**Learning Objectives**
* Describe multi-task learning and transfer learning
* Recognize bias, variance and data-mismatch by looking at the performances of your algorithm on train/dev/test sets

- [Week 2: ML Strategy](#week-2-ml-strategy)
  - [Carrying Out Error Analysis](#carrying-out-error-analysis)
  - [Cleaning Up Incorrectly Labeled Data](#cleaning-up-incorrectly-labeled-data)
    - [Incorrectly labeled examples](#incorrectly-labeled-examples)
  - [Build your First System Quickly, then Iterate](#build-your-first-system-quickly-then-iterate)
    - [Speech recognition example](#speech-recognition-example)
  - [Training and Testing on Different Distributions](#training-and-testing-on-different-distributions)
    - [Cat app example](#cat-app-example)
    - [Speech recognition example](#speech-recognition-example-1)
  - [Bias and Variance with Mismatched Data Distributions](#bias-and-variance-with-mismatched-data-distributions)
    - [Cat classifier example](#cat-classifier-example)
    - [Bias/variance on mismatched training and dev/test sets](#biasvariance-on-mismatched-training-and-devtest-sets)
    - [More general formulation](#more-general-formulation)
  - [Addressing Data Mismatch](#addressing-data-mismatch)
    - [Addressing data mismatch](#addressing-data-mismatch-1)
    - [Artificial data synthesis](#artificial-data-synthesis)
  - [Transfer Learning](#transfer-learning)
    - [Transfer Learning](#transfer-learning-1)
  - [Multi-task Learning](#multi-task-learning)
    - [Simplified autonomous driving examples](#simplified-autonomous-driving-examples)
    - [Neural network architecture](#neural-network-architecture)
    - [When does multi-task learning makes sense?](#when-does-multi-task-learning-makes-sense)
  - [What is End-to-end Deep Learning?](#what-is-end-to-end-deep-learning)
    - [What is end-to-end learning?](#what-is-end-to-end-learning)
    - [Face recognition](#face-recognition)
    - [More examples](#more-examples)
  - [Whether to use end-to-end learning](#whether-to-use-end-to-end-learning)
    - [Pros and cons of end-to-end deep learning](#pros-and-cons-of-end-to-end-deep-learning)


## Carrying Out Error Analysis
If your learning algorithm is not yet at the performance of a human. Then manually examining mistakes that your algorithm is making, can give you insights into what to do next.

Let's say you're working on your cat classifier, and you've achieved 90% accuracy, or equivalently 10% error, on your dev set. 

Your teammate comes to you with a proposal for how to make the algorithm do better, specifically on dogs. You can imagine building a focus effort, maybe to collect more dog pictures, or maybe to design features specific to dogs, or something. In order to make your cat classifier do better on dogs, so it stops misrecognizing these dogs as cats.

So the question is, should you go ahead and start a project focused on the dog problem?

There could be several months of works you could do in order to make your algorithm make fewer mistakes on dog pictures. So is that worth your effort? 

Here's an error analysis procedure that can let you very quickly tell whether or not this could be worth your effort. 
* First, get about, say 100 mislabeled dev set examples
* Then examine them manually. Just count them up one at a time, to see how many of these mislabeled examples in your dev set are actually pictures of dogs. 

Suppose that it turns out that 5% of your 100 mislabeled dev set examples are pictures of dogs. So, that is, if 5 out of 100 of these mislabeled dev set examples are dogs, what this means is that of the 100 examples. Of a typical set of 100 examples you're getting wrong, even if you completely solve the dog problem, you only get 5 out of 100 more correct. 

Or in other words, if only 5% of your errors are dog pictures, then the best you could easily hope to do, if you spend a lot of time on the dog problem. Is that your error might go down from 10% error, down to 9.5% error. So this a 5% relative decrease in error. 

-> This is not the best use of your time. 

This gives a ceiling or upper bound on how much we could improve performance by working on dog problem. In ML, we call this the ceiling on performance, which means, what is the best case, how well could working on the dog problem help us?

Suppose that we look at your 100 mislabeled dev set examples, you find that 50 of them are actually dog images.

-> 50% of them are dog pictures.

In this case, we are more optimistic about spending time on dog problem. If you actually solve the dog problem, your error would go down from this 10%, down to potentially 5% error. 

You might decide that halving your error could be worth a lot of effort. Focus on reducing the problem of mislabeled dogs.

I know that in machine learning, sometimes we speak disparagingly of hand engineering things, or using too much value insight. But if you're building applied systems, then this simple counting procedure, error analysis, can save you a lot of time in terms of deciding what's the most important, what is the most promising direction to focus on.

In fact, if you're looking at 100 mislabeled dev set examples, maybe this is a 5 to 10 minute effort. To manually go through 100 examples, and manually count up how many of them are dogs. And depending on the outcome, whether there's more like 5%, or 50%, or something else. This, in just 5 to 10 minutes, gives you an estimate of how worthwhile this direction is. And could help you make a much better decision, whether or not to spend the next few months focused on trying to find solutions to solve the problem of mislabeled dogs.

In this slide, we'll describe using error analysis to evaluate whether or not a single idea, dogs in this case, is worth working on. 

![alt text](_assets/LookDevExamples.png)
Sometimes you can also evaluate multiple ideas in parallel doing error analysis.

For example, let's say you have several ideas in improving your cat detector.
* Maybe you can improve performance on dogs? 
* Maybe you notice that sometimes, what are called great cats, such as lions, panthers, cheetahs, and so on. That they are being recognized as small cats, or house cats. So you could maybe find a way to work on that. 
* Maybe you find that some of your images are blurry, and it would be nice if you could design something that just works better on blurry images. 

If carrying out error analysis to evaluate these three ideas, what I would do is create a table like this. On the left side, this goes through the set of images you plan to look at manually. So this maybe goes from 1 to 100, if you look at 100 pictures. And the columns of this table, of the spreadsheet, will correspond to the ideas you're evaluating. So the dog problem, the problem of great cats, and blurry images. Last column is for Comments

So remember, during error analysis, you're just looking at dev set examples that your algorithm has misrecognized. So if you find that the first misrecognized image is a picture of a dog, then I'd put a check mark there. And to help myself remember these images, sometimes I'll make a note in the comments. So maybe that was a pit bull picture. If the second picture was blurry, then make a note there. If the third one was a lion, on a rainy day, in the zoo that was misrecognized. Then that's a great cat, and the blurry data. Make a note in the comment section, rainy day at zoo, and it was the rain that made it blurry, and so on.

Then finally, having gone through some set of images, I would count up what percentage of these algorithms. Or what percentage of each of these error categories were attributed to the dog, or great cat, blurry categories. So maybe 8% of these images you examine turn out be dogs, and maybe 43% great cats, and 61% were blurry. So this just means going down each column, and counting up what percentage of images have a check mark in that column. 

![alt text](_assets/EvaluateIdeas.png)

As you're part way through this process, sometimes you notice other categories of mistakes. For example, you might find that Instagram style filter, those fancy image filters, are also messing up your classifier. In that case, it's actually okay, part way through the process, to add another column like that. For the multi-colored filters, the Instagram filters, and the Snapchat filters. And then go through and count up those as well, and figure out what percentage comes from that new error category. 

The conclusion of this process gives you an estimate of how worthwhile it might be to work on each of these different categories of errors.

For example, clearly in this example, a lot of the mistakes were made on blurry images, and quite a lot on were made on great cat images. And so the outcome of this analysis is not that you must work on blurry images. This doesn't give you a rigid mathematical formula that tells you what to do, but it gives you a sense of the best options to pursue.

It also tells you, for example, that no matter how much better you do on dog images, or on Instagram images, you at most improve performance by maybe 8%, or 12%, in these examples. Whereas you can to better on great cat images, or blurry images, the potential improvement. There's a ceiling in terms of how much you could improve performance, is much higher.

Depending on how many ideas you have for improving performance on great cats, on blurry images. Maybe you could pick one of the two, or if you have enough personnel on your team, maybe you can have two different teams. Have one work on improving errors on great cats, and a different team work on improving errors on blurry images.

This quick counting procedure, which you can often do in, at most, small numbers of hours. Can really help you make much better prioritization decisions, and understand how promising different approaches are to work on.

To summarize, to carry out error analysis, you should find a set of mislabeled examples, either in your dev set, or in your development set. And look at the mislabeled examples for false positives and false negatives. And just count up the number of errors that fall into various different categories. During this process, you might be inspired to generate new categories of errors, like we saw. If you're looking through the examples and you say gee, there are a lot of Instagram filters, or Snapchat filters, they're also messing up my classifier. You can create new categories during that process. But by counting up the fraction of examples that are mislabeled in different ways, often this will help you prioritize. Or give you inspiration for new directions to go in. 

## Cleaning Up Incorrectly Labeled Data
The data for your supervised learning problem comprises input X and output labels Y. What if you going through your data and you find that some of these output labels Y are incorrect, you have data which is incorrectly labeled? Is it worth your while to go in to fix up some of these labels? 
### Incorrectly labeled examples
In the cat classification problem, Y equals one for cats and zero for non cats. 

![alt text](_assets/IncorrectLabel.png)

There is an incorrect label.

So I've used the term, mislabeled examples, to refer to if your learning algorithm outputs the wrong value of Y. But I'm going to say, incorrectly labeled examples, to refer to if in the data set you have in the training set or the dev set or the test set, the label for Y, whatever a human label assigned to this piece of data, is actually incorrect.

That's actually a dog so that Y really should have been zero. But maybe the labeler got that one wrong.

If you find that your data has some incorrectly labeled examples, what should you do? 

First, let's consider the training set. 

DL algorithms are quite robust to random errors in the training set.

So long as your errors or your incorrectly labeled examples, so long as those errors are not too far from random, maybe sometimes the labeler just wasn't paying attention or they accidentally, randomly hit the wrong key on the keyboard. If the errors are reasonably random, then it's probably okay to just leave the errors as they are and not spend too much time fixing them.

There's certainly no harm to going into your training set and re-examining the labels and fixing them. Sometimes that is worth doing but your effort might be okay even if you don't. So long as the total data set size is big enough and the actual percentage of errors is maybe not too high. 

I see a lot of machine learning algorithms that trained even when we know that there are few X mistakes in the training set labels and usually works okay. 

There is one caveat to this which is that deep learning algorithms are robust to random errors. They are less robust to systematic errors.

For example, if your labeler consistently labels white dogs as cats, then that is a problem because your classifier will learn to classify all white colored dogs as cats. But random errors or near random errors are usually not too bad for most deep learning algorithms.

This discussion has focused on what to do about incorrectly labeled examples in your training set. How about incorrectly labeled examples in your dev set or test set?

If you're worried about the impact of incorrectly labeled examples on your dev set or test set, what I recommend you do is during error analysis to add one extra column so that you can also count up the number of examples where the label Y was incorrect. 

![alt text](_assets/ErrAnalysis.png)

For example, maybe when you count up the impact on a 100 mislabeled dev set examples, so you're going to find a 100 examples where your classifier's output disagrees with the label in your dev set. And sometimes for a few of those examples, your classifier disagrees with the label because the label was wrong, rather than because your classifier was wrong.

So maybe in this example, you find that the labeler missed a cat in the background. So put the check mark there to signify that example 98 had an incorrect label.

And maybe for this one, the picture is actually a picture of a drawing of a cat rather than a real cat. Maybe you want the labeler to have labeled that Y equals zero rather than Y equals one. And so put another check mark there.

Just as you count up the percent of errors due to other categories like we saw in the previous video, you'd also count up the fraction of percentage of errors due to incorrect labels. Where the Y value in your dev set was wrong, and that accounted for why your learning algorithm made a prediction that differed from what the label on your data says.

So the question now is, is it worthwhile going in to try to fix up this 6% of incorrectly labeled examples?

My advice is:
* If it makes a significant difference to your ability to evaluate algorithms on your dev set, then go ahead and spend the time to fix incorrect labels. 
* If it doesn't make a significant difference to your ability to use the dev set to evaluate classifiers, then it might not be the best use of your time.

For example, three numbers I recommend you look at to try to decide if it's worth going in and reducing the number of mislabeled examples are the following. 

* Overall dev set error, i.e, 10% (90% overall accuracy)
* Errors due incorrect labels, i.e, 0.6%
* Errors due to other causes, i.e, 9.4% (10-0.6=9.4)

In this case, I would say there's 9.4% worth of error that you could focus on fixing, whereas the errors due to incorrect labels is a relatively small fraction of the overall set of errors. So by all means, go in and fix these incorrect labels if you want but it's maybe not the most important thing to do right now.

Let's take another example. Suppose you've made a lot more progress on your learning problem. So instead of 10% error, let's say you brought the errors down to 2%, but still 0.6% of your overall errors are due to incorrect labels.

![alt text](_assets/ErrAnalysisExp.png)

If you want to examine a set of mislabeled dev set images, set that comes from just 2% of dev set data you're mislabeling, then a very large fraction of them, 0.6 divided by 2%, so that is actually 30% rather than 6% of your labels. Your incorrect examples are actually due to incorrectly label examples. And so errors due to other causes are now 1.4%.

When such a high fraction of your mistakes as measured on your dev set due to incorrect labels, then it maybe seems much more worthwhile to fix up the incorrect labels in your dev set. 

If you remember the goal of the dev set, the main purpose of the dev set is, you want to really use it to help you select between two classifiers A and B. So if you're trying out two classifiers A and B, and one has 2.1% error and the other has 1.9% error on your dev set. But you don't trust your dev set anymore to be correctly telling you whether this classifier is actually better than this because your 0.6% of these mistakes are due to incorrect labels. Then there's a good reason to go in and fix the incorrect labels in your dev set. Because in this example on the right is just having a very large impact on the overall assessment of the errors of the algorithm, whereas in the example on the left, the percentage impact is having on your algorithm is still smaller.

Now, if you decide to go into your dev set and manually re-examine the labels and try to fix up some of the labels, here are a few additional guidelines or principles to consider. 
1. Apply whatever process you apply to both your dev and test sets at the same time. We've talk previously about why you want your dev and test sets to come from the same distribution. The dev set is telling you where to aim to target and when you hit it, you want that to generalize to the test set. So your team really works more efficiently to dev and test sets come from the same distribution. So if you're going in to fix something on the dev set, I would apply the same process to the test set to make sure that they continue to come from the same distribution. So we hire someone to examine the labels more carefully. Do that for both your dev and test sets. 
2. Consider examining examples your algorithm got right as well as ones it got wrong. It is easy to look at the examples your algorithm got wrong and just see if any of those need to be fixed. But it's possible that there are some examples that you haven't got right, that should also be fixed. And if you only fix ones that your algorithms got wrong, you end up with more bias estimates of the error of your algorithm. It gives your algorithm a little bit of an unfair advantage. If you just try to double check what it got wrong but you don't also double check what it got right because it might have gotten something right, that it was just lucky on fixing the label would cause it to go from being right to being wrong, on that example. The second bullet isn't always easy to do, so it's not always done. The reason it's not always done is because if you classifier's very accurate, then it's getting fewer things wrong than right. So if your classifier has 98% accuracy, then it's getting 2% of things wrong and 98% of things right. So it's much easier to examine and validate the labels on 2% of the data and it takes much longer to validate labels on 98% of the data, so this isn't always done. That's just something to consider. 
3. Finally, if you go into a dev and test data to correct some of the labels there, you may or may not decide to go and apply the same process for the training set. Remember we said that at the start of this video that it's actually less important to correct the labels in your training set. And it's quite possible you decide to just correct the labels in your dev and test set which are also often smaller than a training set and you might not invest all that extra effort needed to correct the labels in a much larger training set. This is actually okay. We'll talk later this week about some processes for handling when your training data is different in distribution than you dev and test data. Learning algorithms are quite robust to that. It's super important that your dev and test sets come from the same distribution. But if your training set comes from a slightly different distribution, often that's a pretty reasonable thing to do. 

I'd like to wrap up with just a couple of pieces of advice. 
* First, deep learning researchers sometimes like to say things like, "I just fed the data to the algorithm. I trained in and it worked." There is a lot of truth to that in the deep learning era. There is more of feeding data to an algorithm and just training it and doing less hand engineering and using less human insight. But I think that in building practical systems, often there's also more manual error analysis and more human insight that goes into the systems than sometimes deep learning researchers like to acknowledge. 
* Second is that somehow I've seen some engineers and researchers be reluctant to manually look at the examples. Maybe it's not the most interesting thing to do, to sit down and look at a 100 or a couple hundred examples to counter the number of errors. But this is something that I so do myself. When I'm leading a machine learning team and I want to understand what mistakes it is making, I would actually go in and look at the data myself and try to counter the fraction of errors. And I think that because these minutes or maybe a small number of hours of counting data can really help you prioritize where to go next. I find this a very good use of your time and I urge you to consider doing it if you've built a machine learning system and you're trying to decide what ideas or what directions to prioritize things.

## Build your First System Quickly, then Iterate
If you're working on a brand new machine learning application, one of the pieces of advice I often give people is that, I think you should build your first system quickly and then iterate. 

### Speech recognition example
I've worked on speech recognition for many years. And if you're thinking of building a new speech recognition system, there's actually a lot of directions you could go in and a lot of things you could prioritize.

For example:
* There are specific techniques for making speech recognition systems more robust to noisy background. And noisy background could mean cafe noise, like a lot of people talking in the background or car noise, the sounds of cars and highways or other types of noise. There are ways to make a speech recognition system more robust to accented speech. 
* There are ways to make a speech recognition system more robust to accented speech. 
* There are specific problems associated with speakers that are far from the microphone, this is called far-field speech recognition. 
* Young children speech poses special challenges, both in terms of how they pronounce individual words as well as their choice of words and the vocabulary they tend to use. 
* And if sometimes the speaker stutters or if they use nonsensical phrases like oh, ah, um, there are different choices and different techniques for making the transcript that you output, still read more fluently.

For almost any machine learning application, there could be 50 different directions you could go in and each of these directions is reasonable and would make your system better. But the challenge is, how do you pick which of these to focus on. 

What I would recommend you do, if you're starting on building a brand new machine learning application, is to build your first system quickly and then iterate. What I mean by that is I recommend that you:
* Quickly set up a dev/test set and metric. So this is really deciding where to place your target. And if you get it wrong, you can always move it later, but just set up a target somewhere.
* Then I recommend you build an initial machine learning system quickly. Find the training set, train it and see. Start to see and understand how well you're doing against your dev/test set and your valuation metric. 
* When you build your initial system, you will then be able to use bias/variance analysis which we talked about earlier as well as error analysis which we talked about just in the last several videos, to prioritize the next steps. In particular, if error analysis causes you to realize that a lot of the errors are from the speaker being very far from the microphone, which causes special challenges to speech recognition, then that will give you a good reason to focus on techniques to address this called far-field speech recognition which basically means handling when the speaker is very far from the microphone.

Of all the value of building this initial system, it can be a quick and dirty implementation, don't overthink it, but all the value of the initial system is having some learned system, having some trained system allows you to localize bias/variance, to try to prioritize what to do next, allows you to do error analysis, look at some mistakes, to figure out all the different directions you can go in, which ones are actually the most worthwhile.

To recap, what I recommend you do is build your first system quickly, then iterate.
* This advice applies less strongly if you're working on an application area in which you have significant prior experience.
* It also applies a bit less strongly if there's a significant body of academic literature that you can draw on for pretty much the exact same problem you're building.

For example, there's a large academic literature on face recognition. If you're trying to build a face recognizer, it might be okay to build a more complex system from the get-go by building on this large body of academic literature.

But if you are tackling a new problem for the first time, then I would encourage you to really not overthink or not make your first system too complicated. But, just build something quick and dirty and then use that to help you prioritize how to improve your system.

## Training and Testing on Different Distributions
Deep learning algorithms have a huge hunger for training data. They just often work best when you can find enough label training data to put into the training set. This has resulted in many teams sometimes taking whatever data you can find and just shoving it into the training set just to get it more training data. Even if some of this data, or even maybe a lot of this data, doesn't come from the same distribution as your dev and test data. So in a deep learning era, more and more teams are now training on data that comes from a different distribution than your dev and test sets.

### Cat app example
Let's say that you're building a mobile app where users will upload pictures taken from their cell phones, and you want to recognize whether the pictures that your users upload from the mobile app is a cat or not.

![alt text](_assets/CatAppExample.png)

You can now get two sources of data.
* One which is the distribution of data you really care about, this data from a mobile app like that on the right, which tends to be less professionally shot, less well framed, maybe even blurrier because it's shot by amateur users.
* The other source of data you can get is you can crawl the web and just download a lot of, for the sake of this example, let's say you can download a lot of very professionally framed, high resolution, professionally taken images of cats.

Let's say you don't have a lot of users yet for your mobile app. So maybe you've gotten 10,000 pictures uploaded from the mobile app.

But by crawling the web you can download huge numbers of cat pictures, and maybe you have 200,000 pictures of cats downloaded off the Internet. 

We care about the final system does well on mobile app distribution of images. Because in the end, your users will be uploading pictures like those on the right and you need your classifier to do well on that.

But you now have a bit of a dilemma because you have a relatively small dataset, just 10,000 examples drawn from that distribution. And you have a much bigger dataset that's drawn from a different distribution. There's a different appearance of image than the one you actually want. You don't want to use just those 10,000 images because it ends up giving you a relatively small training set. And using those 200,000 images seems helpful, but the dilemma is this 200,000 images isn't from exactly the distribution you want.

**Option 1**

One thing you can do is put both of these data sets together so you now have 210,000 images. And you can then take the 210,000 images and randomly shuffle them into a train, dev, and test set. 

Let's say for the sake of argument that you've decided that your dev and test sets will be 2,500 examples each. So your training set will be 205,000 examples. 

Now so setting up your data this way has some advantages but also disadvantages. 
* The advantage is that now you're training, dev and test sets will all come from the same distribution, so that makes it easier to manage. 
* But the disadvantage, and this is a huge disadvantage, is that if you look at your dev set, of these 2,500 examples, a lot of it will come from the web page distribution of images, rather than what you actually care about, which is the mobile app distribution of images.

It turns out that of your total amount of data, 200,000, so I'll just abbreviate that 200k, out of 210,000, we'll write that as 210k, that comes from web pages. So all of these 2,500 examples on expectation, I think 2,381 of them will come from web pages. This is on expectation, the exact number will vary around depending on how the random shuttle operation went. But on average, only 119 will come from mobile app uploads.

Remember that setting up your dev set is telling your team where to aim the target. And the way you're aiming your target, you're saying spend most of the time optimizing for the web page distribution of images, which is really not what you want. 

So I would recommend against option one, because this is setting up the dev set to tell your team to optimize for a different distribution of data than what you actually care about. So instead of doing this, I would recommend that you instead take another option, which is the following. 

**Option 2**

The training set, let's say it's still 205,000 images, I would have the training set have all 200,000 images from the web. And then you can, if you want, add in 5,000 images from the mobile app. And then for your dev and test sets would be all mobile app images. 

* The training set will include 200,000 images from the web and 5,000 from the mobile app. 
* The dev set will be 2,500 images from the mobile app
* The test set will be 2,500 images also from the mobile app.

The advantage of this way of splitting up your data into train, dev, and test, is that you're now aiming the target where you want it to be. You're telling your team, my dev set has data uploaded from the mobile app and that's the distribution of images you really care about, so let's try to build a machine learning system that does really well on the mobile app distribution of images. 

The disadvantage, of course, is that now your training distribution is different from your dev and test set distributions. But it turns out that this split of your data into train, dev and test will get you better performance over the long term.

![alt text](_assets/CatDataDistribution.png)

### Speech recognition example
Let's say you're building a brand new product, a speech activated rearview mirror for a car. So this is a real product in China. It's making its way into other countries but you can build a rearview mirror to replace this little thing there, so that you can now talk to the rearview mirror and basically say, dear rearview mirror, please help me find navigational directions to the nearest gas station and it'll deal with it. So this is actually a real product, and let's say you're trying to build this for your own country. 

So how can you get data to train up a speech recognition system for this product?

Maybe you've worked on speech recognition for a long time so you have a lot of data from other speech recognition applications, just not from a speech activated rearview mirror. 

Here's how you could split up your training and your dev and test sets.
* So for your training, you can take all the speech data you have that you've accumulated from working on other speech problems, such as 
  * Data you purchased over the years from various speech recognition data vendors. And today you can actually buy data from vendors of x, y pairs, where x is an audio clip and y is a transcript. 
  * Or maybe you've worked on smart speakers, smart voice activated speakers, so you have some data from that. 
  * Maybe you've worked on voice activated keyboards and so on. 

And for the sake of argument, maybe you have 500,000 utterences from all of these sources. 

* And for your dev and test set, maybe you have a much smaller data set that actually came from a speech activated rearview mirror. Because users are asking for navigational queries or trying to find directions to various places. This data set will maybe have a lot more street addresses. Please help me navigate to this street address, or please help me navigate to this gas station. So this distribution of data will be very different than these on the left. 
  * But this is really the data you care about, because this is what you need your product to do well on, so this is what you set your dev and test set to be. 

So what you do in this example is set 
* Your training set to be the 500,000 utterances on the left, and t
* Then your dev and test sets which I'll abbreviate D and T, these could be maybe 10,000 utterances each. That's drawn from actual the speech activated rearview mirror. 

Or alternatively, if you think you don't need to put all 20,000 examples from your speech activated rearview mirror into the dev and test sets, maybe you can take half of that and put that in the training set. 

* So then the training set could be 510,000 utterances, including all 500 from there and 10,000 from the rearview mirror. 
* And then the dev and test sets could maybe be 5,000 utterances each. So of the 20,000 utterances, maybe 10k goes into the training set and 5k into the dev set and 5,000 into the test set. 

So this would be another reasonable way of splitting your data into train, dev, and test. And this gives you a much bigger training set, over 500,000 utterances, than if you were to only use speech activated rearview mirror data for your training set. 

![alt text](_assets/SpeechRecognition.png)

## Bias and Variance with Mismatched Data Distributions
### Cat classifier example
Let's keep using our cat classification example and let's say humans get near perfect performance on this. 

So, Bayes error, or Bayes optimal error, we know is nearly 0% on this problem. 

To carry out error analysis you usually look at the training error and also look at the error on the dev set. 

Let's say, in this example that your training error is 1%, and your dev error is 10%. 

* If your dev data came from the same distribution as your training set, you would say that here you have a large variance problem, that your algorithm's just not generalizing well from the training set which it's doing well on to the dev set, which it's suddenly doing much worse on. 
* But in the setting where your training data and your dev data comes from a different distribution, you can no longer safely draw this conclusion. In particular, maybe it's doing just fine on the dev set, it's just that the training set was really easy because it was high res, very clear images, and maybe the dev set is just much harder. So maybe there isn't a variance problem and this just reflects that the dev set contains images that are much more difficult to classify accurately.

So the problem with this analysis is that when you went from the training error to the dev error, two things changed at a time.
* One is that the algorithm saw data in the training set but not in the dev set. 
* Two, the distribution of data in the dev set is different. 

Because you changed two things at the same time, it's difficult to know of this 9% increase in error, how much of it is because the algorithm didn't see the data in the dev set, so that's some of the variance part of the problem. And how much of it, is because the dev set data is just different. 

In order to tease out these two effects it will be useful to define a new piece of data which we'll call the training-dev set. So, this is a new subset of data, which we carve out that should have the same distribution as training sets, but you don't explicitly train your neural network on this.

![alt text](_assets/CatClassifierExample1.png)

Previously we had set up some training sets and some dev sets and some test sets as follows. 

![alt text](_assets/CatClassifierExample2.png)

The dev and test sets have the same distribution, but the training sets will have some different distribution. 

What we're going to do is randomly shuffle the training sets and then carve out just a piece of the training set to be the training-dev set. 

**Just as the dev and test set have the same distribution, the training set and the training-dev set, also have the same distribution.**

But, the difference is that now you train your neural network, just on the training set proper. You won't let the neural network, you won't run that obligation on the training-dev portion of this data.

To carry out error analysis, what you should do is now look at the error of your classifier on the training set, on the training-dev set, as well as on the dev set. 

So let's say in this example that 
* Your training error is 1%. 
* Let's say the error on the training-dev set is 9%.
* The error on the dev set is 10%, same as before. 

What you can conclude from this is that when you went from training data to training dev data the error really went up a lot. And only the difference between the training data and the training-dev data is that your neural network got to sort the first part of training set. It was trained explicitly on training set, but it wasn't trained explicitly on the training-dev data. 

So this tells you that you have a variance problem. Because the training-dev error was measured on data that comes from the same distribution as your training set. So you know that even though your neural network does well in a training set, it's just not generalizing well to data in the training-dev set which comes from the same distribution, but it's just not generalizing well to data from the same distribution that it hadn't seen before. 

In this example we have really a variance problem. 

Let's look at a different example. 
* Let's say the training error is 1%
* The training-dev error is 1.5%
* The dev set error is 10%.

So now, you have actually a pretty low variance problem, because when you went from training data that you've seen to the training-dev data that the neural network has not seen, the error increases only a little bit, but then it really jumps when you go to the dev set. So this is a data mismatch problem, where data mismatched. Because your learning algorithm was not trained explicitly on data from training-dev or dev, but these two data sets come from different distributions. But whatever algorithm it's learning, it works great on training-dev but it doesn't work well on dev. 

So somehow your algorithm has learned to do well on a different distribution than what you really care about, so we call that a data mismatch problem.

![alt text](_assets/CatClassifierExample3.png)

Let's just look at a few more examples. 
* Let's say that training error is 10%
* Training-dev error is 11%
* Dev error is 12%. 
* Remember that human level proxy for Bayes error is roughly 0%. 

So if you have this type of performance, then you really have a bias, an avoidable bias problem, because you're doing much worse than human level. 

And one last example. 
* If your training error is 10%
* Your training-dev error is 11% 
* Dev error is 20 %

Then it looks like this actually has two issues. 
* One, the avoidable bias is quite high, because you're not even doing that well on the training set. Humans get nearly 0% error, but you're getting 10% error on your training set. 
* The variance here seems quite small, but this data mismatch is quite large. 

So for for this example I will say, you have a large bias or avoidable bias problem as well as a data mismatch problem. 

![alt text](_assets/CatClassifierExample4.png)

### Bias/variance on mismatched training and dev/test sets
The key quantities I would look at are 
* Human level error
* Training set error
* Training-dev set error. So that's the same distribution as the training set, but you didn't train explicitly on it.
* Dev set error

Depending on the differences between these errors, you can get a sense of how big is the avoidable bias, the variance, the data mismatch problems. 

So let's say that 
* Human level error is 4%. 
* Training error is 7%. 
* Training-dev error is 10%. 
* Dev error is 12%
* Test error is 12%

- Human level error and training error gives you a sense of the avoidable bias, because you'd like your algorithm to do at least as well or approach human level performance maybe on the training set.
- Training error and Training-dev error gives a sense of the variance. So how well do you generalize from the training set to the training-dev set? 
- Training-dev error and Dev error gives the sense of how much of a data mismatch problem have you have. 

Technically you could also add one more thing, which is the test set performance, and we'll write test error. You shouldn't be doing development on your test set because you don't want to overfit your test set.

- Gap between Dev error and Test error tells you the degree of overfitting to the dev set. 
  - So if there's a huge gap between your dev set performance and your test set performance, it means you maybe overtuned to the dev set. And so maybe you need to find a bigger dev set. 
  - So remember that your dev set and your test set come from the same distribution. So the only way for there to be a huge gap here, for it to do much better on the dev set than the test set, is if you somehow managed to overfit the dev set. 
  - If that's the case, what you might consider doing is going back and just getting more dev set data. 

![alt text](_assets/MismatchedSets.png)

Now, I've written these numbers, as you go down the list of numbers, always keep going up. Here's one example of numbers that doesn't always go up
* Human level performance is 4%
* Training error is 7%
* Training-dev error is 10%
* Let's say that we go to the dev set. You find that you actually, surprisingly, do much better on the dev set. Maybe this is 6%
* Test error is 6% as well. 

So you have seen effects like this, working on for example a speech recognition task, where the training data turned out to be much harder than your dev set and test set. 

So Training set error and Train-dev set error were evaluated on your training set distribution

Dev error and Test error were evaluated on your dev/test set distribution. 

Sometimes if your dev/test set distribution is much easier for whatever application you're working on then these numbers can actually go down.

![alt text](_assets/MismatchedSets2.png)

If you see funny things like this, there's an even more general formulation of this analysis that might be helpful. 

### More general formulation
Let me motivate this using the speech activated rear-view mirror example. 

It turns out that the numbers we've been writing down can be placed into a table where on the horizontal axis, I'm going to place different data sets. So for example, you might have data from
* General speech recognition task. So you might have a bunch of data that you just collected from a lot of speech recognition problems you worked on from small speakers, data you have purchased and so on.
* Then you all have the rear view mirror specific speech data, recorded inside the car.

So on this x axis on the table, I'm going to vary the data set. 

On the other axis, I'm going to label different ways or algorithms for examining the data. 
* So first, there's human level performance, which is how accurate are humans on each of these data sets? 
* Then there is the error on the examples that your neural network has trained on. 
* And then finally there's error on the examples that your neural network has not trained on. 

Turns out that what we're calling on a human level on the previous slide, there's the number that goes in this box, which is how well do humans do on this category of data. Say data from all sorts of speech recognition tasks, the 500,000 utterances that you could into your training set. And the example in the previous slide is this 4%. 

This number here was our, maybe the training error. Which in the example in the previous slide was 7%, if you're learning algorithm has seen this example, performed gradient descent on this example, and this example came from your training set distribution, or some general speech recognition distribution. How well does your algorithm do on the example it has trained on? 

Then here is the training-dev set error. It's usually a bit higher, which is for data from this distribution, from general speech recognition, if your algorithm did not train explicitly on some examples from this distribution, how well does it do? And that's what we call the training dev error.

![alt text](_assets/MoreGeneral.png)

And then if you move over to the right, this box here is the dev set error, or maybe also the test set error. Which was 6% in the example just now. And dev and test error, it's actually technically two numbers, but either one could go into this box here. And this is if you have data from your rearview mirror, from actually recorded in the car from the rearview mirror application, but your neural network did not perform back propagation on this example, what is the error? 

So what we're doing in the analysis in the previous slide was look at differences between these two numbers, these two numbers, and these two numbers.

![alt text](_assets/MoreGeneral2.png)

![alt text](_assets/MoreGeneral3.png)

And it turns out that it could be useful to also throw in the remaining two entries in this table. And so if this turns out to be also 6%, and the way you get this number is you ask some humans to label their rearview mirror speech data and just measure how good humans are at this task. And maybe this turns out also to be 6%. 

![alt text](_assets/MoreGeneral4.png)

And the way you do that is you take some rearview mirror speech data, put it in the training set so the neural network learns on it as well, and then you measure the error on that subset of the data. But if this is what you get, then, well, it turns out that you're actually already performing at the level of humans on this rearview mirror speech data, so maybe you're actually doing quite well on that distribution of data.

When you do this more sophisticated analysis, it doesn't always give you one clear path forward, but sometimes it just gives you additional insights as well. 

So for example, comparing these two Human level numbers in this case tells us that for humans, the rearview mirror speech data is actually harder than for general speech recognition, because humans get 6% error, rather than 4% error. 

![alt text](_assets/MoreGeneral5.png)

But then looking at these differences as well may help you understand bias and variance and data mismatch problems in different degrees.

This more general formulation is something I've used a few times. I've not used it, but for a lot of problems you find that examining this subset of entries, kind of looking at this difference and this difference and this difference, that that's enough to point you in a pretty promising direction. 

![alt text](_assets/MoreGeneral6.png)

But sometimes filling out this whole table can give you additional insights. 

## Addressing Data Mismatch
If your training set comes from a different distribution, than your dev and test set, and if error analysis shows you that you have a data mismatch problem, what can you do? 

There aren't completely systematic solutions to this, but let's look at some things you could try. 

### Addressing data mismatch
If I find that I have a large data mismatch problem, what I usually do is
* Carry out manual error analysis and try to understand the differences between the training set and the dev/test sets. To avoid overfitting the test set, technically for error analysis, you should manually only look at a dev set and not at a test set. 
  * But as a concrete example, if you're building the speech-activated rear-view mirror application, you might look or, I guess if it's speech, listen to examples in your dev set to try to figure out how your dev set is different than your training set. 
  * For example, you might find that a lot of dev set examples are very noisy and there's a lot of car noise. And this is one way that your dev set differs from your training set. 
  * And maybe you find other categories of errors. For example, in the speech-activated rear-view mirror in your car, you might find that it's often mis-recognizing street numbers because there are a lot more navigational queries which will have street addresses. So, getting street numbers right is really important. 
* When you have insight into the nature of the dev set errors, or you have insight into how the dev set may be different or harder than your training set, what you can do is then try to find ways to make the training data more similar. Or, alternatively, try to collect more data similar to your dev and test sets. 
  * For example, if you find that car noise in the background is a major source of error, one thing you could do is simulate noisy in-car data.
  * Or you find that you're having a hard time recognizing street numbers, maybe you can go and deliberately try to get more data of people speaking out numbers and add that to your training set. 

Now, I realize that this slide is giving a rough guideline for things you could try. This isn't a systematic process and, I guess, it's no guarantee that you get the insights you need to make progress. But I have found that this manual insight, together we're trying to make the data more similar on the dimensions that matter that this often helps on a lot of the problems. 

So, if your goal is to make the training data more similar to your dev set, what are some things you can do? 

One of the techniques you can use is artificial data synthesis and let's discuss that in the context of addressing the car noise problem. 

### Artificial data synthesis
To build a speech recognition system, maybe you don't have a lot of audio that was actually recorded inside the car with the background noise of a car, background noise of a highway, and so on. 

![alt text](_assets/ArtificialDataSynthesis.png)

Let's say that you've recorded a large amount of clean audio without this car background noise.

The quick brown fox jumps over the lazy dog. 

By the way, this sentence is used a lot in AI for testing because this is a short sentence that contains every alphabet from A to Z, so you see this sentence a lot. 

But, given that recording of "the quick brown fox jumps over the lazy dog," you can then also get a recording of car noise. 

If you take these two audio clips and add them together, you can then synthesize what saying "the quick brown fox jumps over the lazy dog" would sound like, if you were saying that in a noisy car.

In practice, you might synthesize other audio effects like reverberation which is the sound of your voice bouncing off the walls of the car and so on. 

But through artificial data synthesis, you might be able to quickly create more data that sounds like it was recorded inside the car without needing to go out there and collect tons of data, maybe thousands or tens of thousands of hours of data in a car that's actually driving along.

If your error analysis shows you that you should try to make your data sound more like it was recorded inside the car, then this could be a reasonable process for synthesizing that type of data to give you a learning algorithm.

There is one note of caution I want to sound on artificial data synthesis which is that, let's say, you have 10,000 hours of data that was recorded against a quiet background. 

And, let's say, that you have just one hour of car noise. 

So, one thing you could try is take this one hour of car noise and repeat it 10,000 times in order to add to this 10,000 hours of data recorded against a quiet background. If you do that, the audio will sound perfectly fine to the human ear, but there is a chance, there is a risk that your learning algorithm will over fit to the one hour of car noise. 

![alt text](_assets/ArtificialData.png)

And, in particular, if this is the set of all audio that you could record in the car or, maybe the sets of all car noise backgrounds you can imagine, if you have just one hour of car noise background, you might be simulating just a very small subset of this space. You might be just synthesizing from a very small subset of this space. 

![alt text](_assets/CarNoise.png)

And to the human ear, all this audio sounds just fine because one hour of car noise sounds just like any other hour of car noise to the human ear. 

But, it's possible that you're synthesizing data from a very small subset of this space, and the neural network might be overfitting to the one hour of car noise that you may have. 

I don't know if it will be practically feasible to inexpensively collect 10,000 hours of car noise so that you don't need to repeat the same one hour of car noise over and over but you have 10,000 unique hours of car noise to add to 10,000 hours of unique audio recording against a clean background. But it's possible, no guarantees. But it is possible that using 10,000 hours of unique car noise rather than just one hour, that could result in better performance for your learning algorithm. 

The challenge with artificial data synthesis is to the human ear, as far as your ears can tell, these 10,000 hours all sound the same as this one hour, so you might end up creating this very impoverished synthesized data set from a much smaller subset of the space without actually realizing it. 

![alt text](_assets/ArtificialData2.png)

Here's another example of artificial data synthesis. 

Let's say you're building a self driving car and so you want to really detect vehicles like this and put a bounding box around it let's say.

![alt text](_assets/CarRecognition.png)

So, one idea that a lot of people have discussed is, well, why should you use computer graphics to simulate tons of images of cars? 

And, in fact, here are a couple of pictures of cars that were generated using computer graphics. And I think these graphics effects are actually pretty good and I can imagine that by synthesizing pictures like these, you could train a pretty good computer vision system for detecting cars. 

![alt text](_assets/ArtificialCar.png)

Unfortunately, the picture that I drew on the previous slide again applies in this setting. 

Maybe this is the set of all cars and, if you synthesize just a very small subset of these cars, then to the human eye, maybe the synthesized images look fine. But you might overfit to this small subset you're synthesizing.

![alt text](_assets/CarSynthesis.png)

In particular, one idea that a lot of people have independently raised is, once you find a video game with good computer graphics of cars and just grab images from them and get a huge data set of pictures of cars, it turns out that if you look at a video game, if the video game has just 20 unique cars in the video game, then the video game looks fine because you're driving around in the video game and you see these 20 other cars and it looks like a pretty realistic simulation. 

But the world has a lot more than 20 unique designs of cars, and if your entire synthesized training set has only 20 distinct cars, then your neural network will probably overfit to these 20 cars. And it's difficult for a person to easily tell that, even though these images look realistic, you're really covering such a tiny subset of the sets of all possible cars. 

To summarize, if you think you have a data mismatch problem, I recommend you do error analysis, or look at the training set, or look at the dev set to try this figure out, to try to gain insight into how these two distributions of data might differ. And then see if you can find some ways to get more training data that looks a bit more like your dev set. 

One of the ways we talked about is artificial data synthesis. And artificial data synthesis does work. In speech recognition, I've seen artificial data synthesis significantly boost the performance of what were already very good speech recognition system. So, it can work very well. 

But, if you're using artificial data synthesis, just be cautious and bear in mind whether or not you might be accidentally simulating data only from a tiny subset of the space of all possible examples. 

## Transfer Learning
One of the most powerful ideas in deep learning is that sometimes you can take knowledge the neural network has learned from one task and apply that knowledge to a separate task. 

So for example, maybe you could have the neural network learn to recognize objects like cats and then use that knowledge or use part of that knowledge to help you do a better job reading x-ray scans. This is called transfer learning. 

### Transfer Learning
Let's say you've trained your neural network on image recognition. So you first take a neural network and train it on X Y pairs, where X is an image and Y is some object. An image is a cat or a dog or a bird or something else. 

![alt text](_assets/ImageRecognition.png)

If you want to take this neural network and adapt, or we say transfer, what is learned to a different task, such as radiology diagnosis, meaning really reading X-ray scans, what you can do is take this last output layer of the neural network and just delete that and delete also the weights feeding into that last output layer and create a new set of randomly initialized weights just for the last layer and have that now output radiology diagnosis.

So to be concrete, during the first phase of training when you're training on an image recognition task, you train all of the usual parameters for the neural network, all the weights, all the layers and you have something that now learns to make image recognition predictions.
* Having trained that neural network, what you now do to implement transfer learning is swap in a new data set X Y, where now X are radiology images. And Y are the diagnoses you want to predict
* Initialize the last layers' weights. Let's call that $W^{[L]}$ and $b^{[L]}$ randomly. 
* Retrain the neural network on this new data set, on the new radiology data set. 

You have a couple options of how you retrain the neural network with radiology data. 
1. If you have a small radiology dataset, you might want to just retrain the weights of the last layer, just $W^{[L]}$ and $b^{[L]}$, and keep the rest of the parameters fixed. 
2. If you have enough data, you could also retrain all the layers of the rest of the neural network. 

The rule of thumb is maybe 
* If you have a small data set, then just retrain the one last layer at the output layer. Or maybe that last one or two layers. 
* If you have a lot of data, then maybe you can retrain all the parameters in the network. 
  * If you retrain all the parameters in the neural network, then this initial phase of training on image recognition is sometimes called pre-training, because you're using image recognitions data to pre-initialize or really pre-train the weights of the neural network. 
  * If you are updating all the weights afterwards, then training on the radiology data sometimes that's called fine tuning.

If you hear the words pre-training and fine tuning in a deep learning context, this is what they mean when they refer to pre-training and fine tuning weights in a transfer learning source.

What you've done in this example, is you've taken knowledge learned from image recognition and applied it or transferred it to radiology diagnosis. 

The reason this can be helpful is that a lot of the low level features such as detecting edges, detecting curves, detecting positive objects. Learning from that, from a very large image recognition data set, might help your learning algorithm do better in radiology diagnosis. It's just learned a lot about the structure and the nature of how images look like and some of that knowledge will be useful. So having learned to recognize images, it might have learned enough about just what parts of different images look like, that knowledge about lines, dots, curves, and so on, maybe small parts of objects, that knowledge could help your radiology diagnosis network learn a bit faster or learn with less data.

![alt text](_assets/Radiology.png)

Let's say that you've trained a speech recognition system so now X is input of audio or audio snippets, and Y is some ink transcript. So you've trained a speech recognition system to output your transcripts. 

Let's say that you now want to build a "wake words" or a "trigger words" detection system. 

Recall that a wake word or the trigger word are the words we say in order to wake up speech control devices in our houses such as saying "Alexa" to wake up an Amazon Echo or "OK Google" to wake up a Google device or "hey Siri" to wake up an Apple device or saying "Ni hao baidu" to wake up a baidu device.

In order to do this:
* Take out the last layer of the neural network again and create a new output node. 
  * Sometimes another thing you could do is actually create not just a single new output, but actually create several new layers to your neural network to try to predict the labels Y for your wake word detection problem. 
  * Depending on how much data you have, you might just retrain the new layers of the network or maybe you could retrain even more layers of this neural network.

![alt text](_assets/Audio.png)

When does transfer learning make sense? 

Transfer learning makes sense when you have a lot of data for the problem you're transferring from and usually relatively less data for the problem you're transferring to. 

For example:
* Let's say you have a million examples for image recognition task. So that's a lot of data to learn a lot of low level features or to learn a lot of useful features in the earlier layers in neural network. 
* But for the radiology task, maybe you have only a hundred examples. So you have very low data for the radiology diagnosis problem, maybe only 100 x-ray scans. 

A lot of knowledge you learn from image recognition can be transferred and can really help you get going with radiology recognition even if you don't have all the data for radiology.

For speech recognition, maybe you've trained the speech recognition system on 10000 hours of data. So, you've learned a lot about what human voices sounds like from that 10000 hours of data, which really is a lot. 

But for your trigger word detection, maybe you have only one hour of data. So, that's not a lot of data to fit a lot of parameters. So in this case, a lot of what you learn about what human voices sound like, what are components of human speech and so on, that can be really helpful for building a good wake word detector, even though you have a relatively small dataset or at least a much smaller dataset for the wake word detection task.

In both of these cases, you're transferring from a problem with a lot of data to a problem with relatively little data. 

One case where transfer learning would not make sense, is if the opposite was true.

If you had a hundred images for image recognition and you had 100 images for radiology diagnosis or even a thousand images for radiology diagnosis, one would think about it is that to do well on radiology diagnosis, assuming what you really want to do well on this radiology diagnosis, having radiology images is much more valuable than having cat and dog and so on images. So each example here is much more valuable than each example there, at least for the purpose of building a good radiology system.

If you already have more data for radiology, it's not that likely that having 100 images of your random objects of cats and dogs and cars and so on will be that helpful, because the value of one example of image from your image recognition task of cats and dogs is just less valuable than one example of an x-ray image for the task of building a good radiology system.

This would be one example where transfer learning, well, it might not hurt but I wouldn't expect it to give you any meaningful gain either. 

Similarly, if you'd built a speech recognition system on 10 hours of data and you actually have 10 hours or maybe even more, say 50 hours of data for wake word detection, you know it won't, it may or may not hurt, maybe it won't hurt to include that 10 hours of data to your transfer learning, but you just wouldn't expect to get a meaningful gain. 

![alt text](_assets/TransferLearning.png)

To summarize, when does transfer learning make sense? 

If you're trying to learn from some Task A and transfer some of the knowledge to some Task B, then transfer learning makes sense when 
* Task A and B have the same input X. 
  * In the first example, A and B both have images as input. 
  * In the second example, both have audio clips as input.
* It tends to make sense when you have a lot more data for Task A than for Task B. 
  * All this is under the assumption that what you really want to do well on is Task B. 
  * Because data for Task B is more valuable for Task B, usually you just need a lot more data for Task A because you know, each example from Task A is just less valuable for Task B than each example for Task B. 
* If you suspect that low level features from Task A could be helpful for learning Task B. 
  * Learning image recognition teaches you enough about images to have a radiology diagnosis.
  * Learning speech recognition teaches you about human speech to help you with trigger word or wake word detection.

To summarize, transfer learning has been most useful if you're trying to do well on some Task B, usually a problem where you have relatively little data. For example, in radiology, you know it's difficult to get that many x-ray scans to build a good radiology diagnosis system. In that case, you might find a related but different task, such as image recognition, where you can get maybe a million images and learn a lot of load-over features from that, so that you can then try to do well on Task B on your radiology task despite not having that much data for it.

When transfer learning makes sense? 

It does help the performance of your learning task significantly. But I've also seen sometimes seen transfer learning applied in settings where Task A actually has less data than Task B and in those cases, you kind of don't expect to see much of a gain. 

## Multi-task Learning
In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And then each of these task helps hopefully all of the other task. 

### Simplified autonomous driving examples
Let's say you're building an autonomous vehicle, building a self driving car. 

Then your self driving car would need to detect several different things such as pedestrians, detect other cars, detect stop signs. And also detect traffic lights and also other things. 

In this example, there is a stop sign in this image and there is a car in this image but there aren't any pedestrians or traffic lights. 

![alt text](_assets/AutoDriving.png)

So if this image is an input for an example, $X^{(i)}$, then Instead of having one label $y^{(i)}$, you would actually a four labels. 

In this example, there are no pedestrians, there is a car, there is a stop sign and there are no traffic lights. 

And if you try and detect other things, there may be $y^{(i)}$ has even more dimensions. But for now let's stick with these four. 

So $y^{(i)}$ is a 4 by 1 vector. 

If you look at the training test labels as a whole, then similar to before, we'll stack the training data's labels horizontally as follows, $y^{(1)}$ up to $y^{(m)}$.

![alt text](_assets/AutoDriving1.png)

Except that now $y^{(i)}$ is a 4 by 1 vector so each of these is a tall column vector. 

And so this matrix Y is now a 4 by m matrix, whereas previously, when y was single real number, this would have been a 1 by m matrix. 

So what you can do is now train a neural network to predict these values of y. 

### Neural network architecture
You can have a neural network input x and output now a four dimensional value for y. 

![alt text](_assets/NN.png)

Notice here for the output there I've drawn four nodes. 
* The first node when we try to predict is there a pedestrian in this picture. 
* The second output will predict is there a car here
* The third one predicts is there a stop sign and this will
* The forth one predicts maybe is there a traffic light. 

![alt text](_assets/NN1.png)

![alt text](_assets/NN2.png)

So y hat here is four dimensional. 

To train this neural network, you now need to define the loss for the neural network. And so given a predicted output $\hat{y}^{(i)}$ which is 4 by 1 dimensional. The loss averaged over your entire training set would be 

${1 \over m}\Sigma_{i=1}^m \Sigma_{j=1}^4 \ell(\hat{y}^{(i)}_j, y^{(i)}_j)$

So it's just summing over at the four components of pedestrian, car, stop sign, traffic lights. The script $\ell$ is the usual logistic loss. 

$\ell=-y^{(i)}_j log \hat{y}^{(i)}_j - (1-y^{(i)}_j)log(1-\hat{y}^{(i)}_j)$

The main difference compared to the earlier binding classification examples is that you're now summing over j equals 1 through 4.

The main difference between this and softmax regression, is that unlike softmax regression, which assigned a single label to single example. This one image can have multiple labels. So you're not saying that each image is either a picture of a pedestrian, or a picture of car, a picture of a stop sign, picture of a traffic light. You're asking for each picture, does it have a pedestrian, or a car a stop sign or traffic light, and multiple objects could appear in the same image.

In fact, in the example on the previous slide, we had both a car and a stop sign in that image, but no pedestrians and traffic lights. So you're not assigning a single label to an image, you're going through the different classes and asking for each of the classes does that class, does that type of object appear in the image?

That's why I'm saying that with this setting, one image can have multiple labels.

If you train a neural network to minimize this cost function, you are carrying out multi-task learning. 

Because what you're doing is building a single neural network that is looking at each image and basically solving four problems. It's trying to tell you does each image have each of these four objects in it.

One other thing you could have done is just train four separate neural networks, instead of train one network to do four things. 

But if some of the earlier features in neural network can be shared between these different types of objects, then you find that training one neural network to do four things results in better performance than training four completely separate neural networks to do the four tasks separately. So that's the power of multi-task learning. 

One other detail, so far I've described this algorithm as if every image had every single label. It turns out that multi-task learning also works even if some of the images we'll label only some of the objects.

The first training example, let's say someone, your labeler had told you there's a pedestrian, there's no car, but they didn't bother to label whether or not there's a stop sign or whether or not there's a traffic light.

Maybe for the second example, there is a pedestrian, there is a car, but again the labeler, when they looked at that image, they just didn't label it, whether it had a stop sign or whether it had a traffic light, and so on.

Maybe some examples are fully labeled, and maybe some examples, they were just labeling for the presence and absence of cars so there's some question marks, and so on.

With a data set like this, you can still train your learning algorithm to do four tasks at the same time, even when some images have only a subset of the labels and others are sort of question marks or don't cares. 

The way you train your algorithm, even when some of these labels are question marks or really unlabeled is that in this sum over j from 1 to 4, you would sum only over values of j with a 0 or 1 label. 

Whenever there's a question mark, you just omit that term from summation but just sum over only the values where there is a label.That allows you to use datasets like this as well.

![alt text](_assets/NN3.png)

### When does multi-task learning makes sense? 
I'll say it makes sense usually when three things are true. 
* One is if your training on a set of tasks that could benefit from having shared low-level features. So for the autonomous driving example, it makes sense that recognizing traffic lights and cars and pedestrians, those should have similar features that could also help you recognize stop signs, because these are all features of roads.
* Second, this is less of a hard and fast rule, so this isn't always true. But what I see from a lot of successful multi-task learning settings is that the amount of data you have for each task is quite similar. 
  * If you recall from transfer learning, you learn from some task A and transfer it to some task B. So if you have a million examples for task A then and 1,000 examples for task B, then all the knowledge you learned from that million examples could really help augment the much smaller data set you have for task B.
  * Well how about multi-task learning? In multi-task learning you usually have a lot more tasks than just two. So maybe you have, previously we had 4 tasks but let's say you have 100 tasks. And you're going to do multi-task learning to try to recognize 100 different types of objects at the same time. So what you may find is that you may have 1,000 examples per task and so if you focus on the performance of just one task, let's focus on the performance on the 100th task, you can call $A_{100}$. If you are trying to do this final task in isolation, you would have had just a thousand examples to train this one task, this one of the 100 tasks that by training on these 99 other tasks. These in aggregate have 99,000 training examples which could be a big boost, could give a lot of knowledge to argument this otherwise, relatively small 1,000 example training set that you have for task $A_{100}$. Symmetrically, every one of the other 99 tasks can provide some data or provide some knowledge that help every one of the other tasks in this list of 100 tasks.
  * So the second bullet isn't a hard and fast rule but what I tend to look at is if you focus on any one task, for that to get a big boost for multi-task learning, the other tasks in aggregate need to have quite a lot more data than for that one task. And so one way to satisfy that is if a lot of tasks like we have in this example on the right, and if the amount of data you have in each task is quite similar. 
  * The key really is that if you already have 1,000 examples for 1 task, then for all of the other tasks you better have a lot more than 1,000 examples if those other task are meant to help you do better on this final task. 
* Finally multi-task learning tends to make more sense when you can train a big enough neural network to do well on all the tasks. So the alternative to multi-task learning would be to train a separate neural network for each task. So rather than training one neural network for pedestrian, car, stop sign, and traffic light detection, you could have trained one neural network for pedestrian detection, one neural network for car detection, one neural network for stop sign detection, and one neural network for traffic light detection. 
  * What a researcher, Rich Carona, found many years ago was that the only times multi-task learning hurts performance compared to training separate neural networks is if your neural network isn't big enough. 
  * But if you can train a big enough neural network, then multi-task learning certainly should not or should very rarely hurt performance. And hopefully it will actually help performance compared to if you were training neural networks to do these different tasks in isolation. 

![alt text](_assets/MultitaskLearning.png)

In practice, multi-task learning is used much less often than transfer learning. 

I see a lot of applications of transfer learning where you have a problem you want to solve with a small amount of data. So you find a related problem with a lot of data to learn something and transfer that to this new problem.

Multi-task learning is just more rare that you have a huge set of tasks you want to use that you want to do well on, you can train all of those tasks at the same time. Maybe the one example is computer vision. In object detection I see more applications of multi-task learning where one neural network trying to detect a whole bunch of objects at the same time works better than different neural networks trained separately to detect objects. 

I would say that on average transfer learning is used much more today than multi-task learning, but both are useful tools to have in your arsenal.

To summarize, multi-task learning enables you to train one neural network to do many tasks and this can give you better performance than if you were to do the tasks in isolation.

One note of caution, in practice I see that transfer learning is used much more often than multi-task learning. So I do see a lot of tasks where if you want to solve a machine learning problem but you have a relatively small data set, then transfer learning can really help. Where if you find a related problem but you have a much bigger data set, you can train in your neural network from there and then transfer it to the problem where we have very low data. So transfer learning is used a lot today. There are some applications of transfer multi-task learning as well, but multi-task learning I think is used much less often than transfer learning. 

Maybe the one exception is computer vision object detection, where I do see a lot of applications of training a neural network to detect lots of different objects. And that works better than training separate neural networks and detecting the visual objects.

On average I think that even though transfer learning and multi-task learning often you're presented in a similar way, in practice I've seen a lot more applications of transfer learning than of multi-task learning. I think because often it's just difficult to set up or to find so many different tasks that you would actually want to train a single neural network for.Again, with some sort of computer vision, object detection examples being the most notable exception.

## What is End-to-end Deep Learning?
One of the most exciting recent developments in deep learning, has been the rise of end-to-end deep learning.

Briefly, there have been some data processing systems, or learning systems that require multiple stages of processing. And what end-to-end deep learning does, is it can take all those multiple stages, and replace it usually with just a single neural network. 
### What is end-to-end learning?
Take speech recognition as an example, where your goal is to take an input X such an audio clip, and map it to an output Y, which is a transcript of the audio clip.

Traditionally, speech recognition required many stages of processing. 
* Extract some features, some hand-designed features of the audio. 
  * If you've heard of MFCC, that's an algorithm for extracting a certain set of hand designed features for audio. 
* Then having extracted some low level features, you might apply a machine learning algorithm, to find the phonemes in the audio clip. 
  * Phonemes are the basic units of sound. So for example, the word cat is made out of three sounds. The Cu- Ah- and Tu- so they extract those.
* Then you string together phonemes to form individual words. 
* Then you string those together to form the transcripts of the audio clip.

![alt text](_assets/SpeechRecognition1.png)

In contrast to this pipeline with a lot of stages, what end-to-end deep learning does, is you can train a huge neural network to just input the audio clip, and have it directly output the transcript. 

![alt text](_assets/SpeechRecognition2.png)

One interesting sociological effect in AI is that as end-to-end deep learning started to work better, there were some researchers that had spent many years of their career designing individual steps of the pipeline. So there were some researchers in different disciplines not just in speech recognition. Maybe in computer vision, and other areas as well, that had spent a lot of time you know, written multiple papers, maybe even built a large part of their career, engineering features or engineering other pieces of the pipeline.

When end-to-end deep learning just took the last training set and learned the function mapping from x and y directly, really bypassing a lot of these intermediate steps, it was challenging for some disciplines to come around to accepting this alternative way of building AI systems. Because it really obsoleted in some cases, many years of research in some of the intermediate components. 

It turns out that one of the challenges of end-to-end deep learning is that you might need a lot of data before it works well.

For example, if you're training on 3,000 hours of data to build a speech recognition system, then the traditional pipeline, the full traditional pipeline works really well.

It's only when you have a very large data set, you know one could say 10,000 hours of data, anything going up to maybe 100,000 hours of data that the end-to end-approach then suddenly starts to work really well.

So when you have a smaller data set, the more traditional pipeline approach actually works just as well. Often works even better. And you need a large data set before the end-to-end approach really shines.

If you have a medium amount of data, then there are also intermediate approaches where maybe you input audio and bypass the features and just learn to output the phonemes of the neural network, and then at some other stages as well. This will be a step toward end-to-end learning, but not all the way there.

![alt text](_assets/EndToEndLearning.png)

### Face recognition
This is a picture of a face recognition turnstile built by a researcher, Yuanqing Lin at Baidu, where this is a camera and it looks at the person approaching the gate, and if it recognizes the person then, you know the turnstile automatically lets them through.

![alt text](_assets/FaceRecognition.png)

Rather than needing to swipe an RFID badge to enter this facility, in increasingly many offices in China and hopefully more and more in other countries as well, you can just approach the turnstile and if it recognizes your face it just lets you through without needing you to carry an RFID badge.

How do you build a system like this? 

One thing you could do is just look at the image that the camera is capturing.

Maybe this is a camera image, you have someone approaching the turnstile. So this might be the image X that you that your camera is capturing.

![alt text](_assets/FaceRecognition1.png)

One thing you could do is try to learn a function mapping directly from the image X to the identity of the person Y.

It turns out this is not the best approach. And one of the problems is that you know, the person approaching the turnstile can approach from lots of different directions. So they could be green positions, they could be in blue position. You know, sometimes they're closer to the camera, so they appear bigger in the image. Sometimes they're already closer to the camera, so that face appears much bigger.

![alt text](_assets/FaceRecognition2.png)

What it has actually done to build these turnstiles, is not to just take the raw image and feed it to a neural net to try to figure out a person's identity. 

Instead, the best approach to date, seems to be a multi-step approach
* First, you run one piece of software to detect the person's face. So this first detector to figure out where's the person's face. 
* Having detected the person's face, you then zoom in to that part of the image and crop that image so that the person's face is centered.
* Then, it is this picture that I guess I drew here in red.
* This is then fed to the neural network, to then try to learn, or estimate the person's identity.

![alt text](_assets/FaceRecognition3.png)

What researchers have found, is that instead of trying to learn everything on one step, by breaking this problem down into two simpler steps
* First is figure out where is the face. 
* Second, is look at the face and figure out who this actually is. 

This second approach allows the learning algorithm or really two learning algorithms to solve two much simpler tasks and results in overall better performance. 

By the way, if you want to know how step two here actually works, I've actually simplified the description a bit.

The way the second step is actually trained, as you train your neural network, that takes as input two images, and what then your network does is it takes this input two images and it tells you if these two are the same person or not. 

![alt text](_assets/FaceRecognition4.png)

If you then have say 10,000 employees IDs on file, you can then take this image in red, and quickly compare it against maybe all 10,000 employee IDs on file to try to figure out if this picture in red is indeed one of your 10000 employees that you should allow into this facility or that should allow into your office building. This is a turnstile that is giving employees access to a workplace.

Why is it that the two step approach works better? 

There are actually two reasons for that.
* One is that each of the two problems you're solving is actually much simpler.
* Second, is that you have a lot of data for each of the two sub-tasks.
  * In particular, there is a lot of data you can obtain for face detection.
  * For task one, where the task is to look at an image and figure out where is the person's face and the image. There is a lot of label data X, comma Y where X is a picture and y shows the position of the person's face. So you could build a neural network to do task one quite well. 
  * Then separately, there's a lot of data for task two as well. Today, leading companies have let's say, hundreds of millions of pictures of people's faces. So given a closely cropped image, like this red image or this one down here, today leading face recognition teams have at least hundreds of millions of images that they could use to look at two images and try to figure out the identity or to figure out if it's the same person or not. So there's also a lot of data for task two.

![alt text](_assets/FaceRecognition5.png)

In contrast, if you were to try to learn everything at the same time, there is much less data of the form X comma Y. Where X is image like this taken from the turnstile, and Y is the identity of the person.

Because you don't have enough data to solve this end-to-end learning problem, but you do have enough data to solve sub-problems one and two, in practice, breaking this down to two sub-problems results in better performance than a pure end-to-end deep learning approach. 

Although if you had enough data for the end-to-end approach, maybe the end-to-end approach would work better, but that's not actually what works best in practice today. 

### More examples
Take machine translation. 

Traditionally, machine translation systems also had a long complicated pipeline, where you first take say English, text and then do text analysis. Basically, extract a bunch of features off the text, and so on. After many many steps you'd end up with say, a translation of the English text into French.

Because, for machine translation, you do have a lot of pairs of English comma French sentences. End-to-end deep learning works quite well for machine translation. 

![alt text](_assets/MachineTranslation.png)

And that's because today, it is possible to gather large data sets of X-Y pairs where that's the English sentence and that's the corresponding French translation. So in this example, end-to-end deep learning works well. 

One last example, let's say that you want to look at an X-ray picture of a hand of a child, and estimate the age of a child.

You know, when I first heard about this problem, I thought this is a very cool crime scene investigation task where you find maybe tragically the skeleton of a child, and you want to figure out how old the child was.

It turns out that typical application of this problem, estimating age of a child from an X-ray is less dramatic than this crime scene investigation I was picturing. It turns out that pediatricians use this to estimate whether or not a child is growing or developing normally.

A non end-to-end approach to this, would be you look at an image and then you segment out or recognize the bones. So, just try to figure out where is that bone segment? Where is that bone segment? Where is that bone segment? And so on.

Then. Knowing the lengths of the different bones, you can sort of go to a look up table showing the average bone lengths in a child's hand and then use that to estimate the child's age. 

This approach actually works pretty well.

In contrast, if you were to go straight from the image to the child's age, then you would need a lot of data to do that directly and as far as I know, this approach does not work as well today just because there isn't enough data to train this task in an end-to-end fashion.

Whereas in contrast, you can imagine that by breaking down this problem into two steps.
* Step one is a relatively simple problem. Maybe you don't need that much data. Maybe you don't need that many X-ray images to segment out the bones.
* Task two, by collecting statistics of a number of children's hands, you can also get decent estimates of that without too much data.

![alt text](_assets/EstimateChildAge.png)

This multi-step approach seems promising. Maybe more promising than the end-to-end approach, at least until you can get more data for the end-to-end learning approach.

End-to-end deep learning works. It can work really well and it can really simplify the system and not require you to build so many hand-designed individual components. But it's also not panacea, it doesn't always work. 

## Whether to use end-to-end learning
### Pros and cons of end-to-end deep learning
Pros
* End-to-end learning really just lets the data speak.
  * If you have enough X,Y data then whatever is the most appropriate function mapping from X to Y, if you train a big enough neural network, hopefully the neural network will figure it out. By having a pure machine learning approach, your neural network learning input from X to Y may be more able to capture whatever statistics are in the data, rather than being forced to reflect human preconceptions.
  * For example, in the case of speech recognition earlier speech systems had this notion of a phoneme which was a basic unit of sound like C, A, and T for the word cat. 
  * I think that phonemes are an artifact created by human linguists. I actually think that phonemes are a fantasy of linguists that are a reasonable description of language, but it's not obvious that you want to force your learning algorithm to think in phonemes.
  * If you let your learning algorithm learn whatever representation it wants to learn rather than forcing your learning algorithm to use phonemes as a representation, then its overall performance might end up being better.
* There's less hand designing of components needed.
  * This could also simplify your design work flow, that you just don't need to spend a lot of time hand designing features, hand designing these intermediate representations.

Cons:
* May need large amount of data.
  * To learn this X to Y mapping directly, you might need a lot of data of X, Y and we were seeing in a previous video some examples of where you could obtain a lot of data for subtasks. 
  * Such as for face recognition, we could find a lot data for finding a face in the image, as well as identifying the face once you found a face, but there was just less data available for the entire end-to-end task. 
  * So X, this is the input end of the end-to-end learning and Y is the output end. And so you need all the data X Y with both the input end and the output end in order to train these systems, and this is why we call it end-to-end learning value as well because you're learning a direct mapping from one end of the system all the way to the other end of the system. 
* It excludes potentially useful hand designed components.
  * Machine learning researchers tend to speak disparagingly of hand designing things. If you don't have a lot of data, then your learning algorithm doesn't have that much insight it can gain from your data if your training set is small. And so hand designing a component can really be a way for you to inject manual knowledge into the algorithm, and that's not always a bad thing.
  * I think of a learning algorithm as having two main sources of knowledge. One is the data and the other is whatever you hand design, be it components, or features, or other things.
  * When you have a ton of data it's less important to hand design things but when you don't have much data, then having a carefully hand-designed system can actually allow humans to inject a lot of knowledge about the problem into an algorithm deck and that should be very helpful.
  * One of the downsides of end-to-end deep learning is that it excludes potentially useful hand-designed components. And hand-designed components could be very helpful if well designed. They could also be harmful if it really limits your performance, such as if you force an algorithm to think in phonemes when maybe it could have discovered a better representation by itself. So it's kind of a double edged sword that could hurt or help but it does tend to help more, hand-designed components tend to help more when you're training on a small training set

![alt text](_assets/EndToEnd-ProsCons.png)

If you're building a new machine learning system and you're trying to decide whether or not to use end-to-end deep learning, I think the key question is, do you have sufficient data to learn the function of the complexity needed to map from X to Y?

I don't have a formal definition of this phrase, complexity needed, but intuitively, if you're trying to learn a function from X to Y, that is looking at an image like this and recognizing the position of the bones in this image, then maybe this seems like a relatively simple problem to identify the bones of the image and maybe they'll need that much data for that task.

![alt text](_assets/EndToEnd_Apply.png)

Given a picture of a person, maybe finding the face of that person in the image doesn't seem like that hard a problem, so maybe you don't need too much data to find the face of a person. At least maybe you can find enough data to solve that task.

Whereas in contrast, the function needed to look at the hand and map that directly to the age of a child, that seems like a much more complex problem that intuitively maybe you need more data to learn if you were to apply a pure end-to-end deep learning approach.

You may know that I've been spending time helping out an autonomous driving company, Drive.ai. So I'm actually very excited about autonomous driving. So how do you build a car that drives itself?

Here's one thing you could do, and this is not an end-to-end deep learning approach.
* You can take as input an image of what's in front of your car, maybe radar, lidar, other sensor readings as well, but to simplify the description, let's just say you take a picture of what's in front or what's around your car.
* Then to drive your car safely you need to detect other cars and you also need to detect pedestrians. You need to detect other things, of course, but we'll just present a simplified example here.
* Having figured out where are the other cars and pedestrians, you then need to plan your own route. So in other words, if you see where are the other cars, where are the pedestrians, you need to decide how to steer your own car, what path to steer your own car for the next several seconds.
* Having decided that you're going to drive a certain path, then you need to execute this by generating the appropriate steering, as well as acceleration and braking commands.

![alt text](_assets/EndToEnd_Example.png)

In going from your image or your sensory inputs to detecting cars and pedestrians, that can be done pretty well using deep learning, but then having figured out where the other cars and pedestrians are going, to select this route to exactly how you want to move your car, usually that's not to done with deep learning.

Instead that's done with a piece of software called Motion Planning. And if you ever take a course in robotics you'll learn about motion planning. 

Then having decided what's the path you want to steer your car through, there'll be some other algorithm, we're going to say it's a control algorithm that then generates the exact decision, that then decides exactly how much to turn the steering wheel and how much to step on the accelerator or step on the brake. 

I think what this example illustrates is that you want to use machine learning or use deep learning to learn some individual components and when applying supervised learning you should carefully choose what types of X to Y mappings you want to learn depending on what task you can get data for. 

In contrast, it is exciting to talk about a pure end-to-end deep learning approach where you input an image and directly output a steering. But given data availability and the types of things we can learn with neural networks today, this is actually not the most promising approach or this is not an approach that I think teams have gotten to work best.

![alt text](_assets/EndToEnd_DL.png)

I think this pure end-to-end deep learning approach is actually less promising than more sophisticated approaches like this, given the availability of data and our ability to train neural networks today.






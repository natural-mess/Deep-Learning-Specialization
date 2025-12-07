# Week 2: Natural Language Processing & Word Embeddings

**Learning Objectives**
* Explain how word embeddings capture relationships between words
* Load pre-trained word vectors
* Measure similarity between word vectors using cosine similarity
* Use word embeddings to solve word analogy problems such as Man is to Woman as King is to ______.
* Reduce bias in word embeddings
* Create an embedding layer in Keras with pre-trained word vectors
* Describe how negative sampling learns word vectors more efficiently than other methods
* Explain the advantages and disadvantages of the GloVe algorithm
* Build a sentiment classifier using word embeddings
* Build and train a more sophisticated classifier using an LSTM

## Word Representation
### Word representation
Last week, we learned about RNNs, GRUs, and LSTMs. In this week, you see how many of these ideas can be applied to NLP, to Natural Language Processing, which is one of the features of AI because it's really being revolutionized by deep learning.

One of the key ideas you learn about is word embeddings, which is a way of representing words that let your algorithms automatically understand analogies like that, man is to woman, as king is to queen, and many other examples.

Through these ideas of word embeddings, you'll be able to build NLP applications, even with models of size usually relatively small label training sets. Finally towards the end of the week, you'll see how to debias word embeddings that's to reduce undesirable gender or ethnicity or other types of bias that learning algorithms can sometimes pick up.

So far, we've been representing words using a vocabulary of words, and a vocabulary from the previous week might be say, 10,000 words.

V = [a, aaron, ..., zulu, `<UNK>`]

|V| = 10000

We've been representing words using a one-hot vector.

For example, if man is word number 5391 in this dictionary, then you represent him with a vector with 1 in position 5391. And I'm also going to use $O_{5391}$ to represent this factor, where O here stands for one-hot.

Then, if woman is word number 9853, then you represent it with $O_{9853}$ which just has a 1 in position 9853 and 0's elsewhere. 

And then other words king, queen, apple, orange will be similarly represented with one-hot vector.

![alt text](_assets/OneHotExample.png)

One of the weaknesses of this representation is that it treats each word as a thing unto itself, and it doesn't allow an algorithm to easily generalize the cross words.

For example, let's say you have a language model that has learned that when you see "I want a glass of orange __".

Well, what do you think the next word will be? Very likely, it'll be juice.

But even if the learning algorithm has learned that "I want a glass of orange juice" is a likely sentence, if it sees "I want a glass of apple __". As far as it knows the relationship between apple and orange is not any closer as the relationship between any of the other words man, woman, king, queen, and orange. And so, it's not easy for the learning algorithm to generalize from knowing that "orange juice" is a popular thing, to recognizing that "apple juice" might also be a popular thing or a popular phrase.

This is because the product between any 2 different one-hot vectors is 0.

If you take any 2 vectors say, queen and king and take product of them, the end product is 0.

If you take apple and orange and any product of them, the end product is 0.

You couldn't distance between any pair of these vectors is also the same.

So it just doesn't know that somehow apple and orange are much more similar than king and orange or queen and orange.

The Problem With One-Hot Vectors

Traditional NLP represented each word as a one-hot vector (e.g., a 10,000-dimensional vector with a single 1).

Problems:
* All words are equally distant (no similarity).
* Vocabulary size is huge.
* Models cannot generalize.
* "Apple" and "Orange" look as unrelated as "Apple" and "Airplane".

![alt text](_assets/WordRepresentation.png)

### Featurized representation: word embedding
So, won't it be nice if instead of a one-hot presentation we can instead learn a featurized representation with each of these words, a man, woman, king, queen, apple, orange or really for every word in the dictionary, we could learn a set of features and values for each of them.

For example, each of these words, we want to know what is the gender associated with each of these things.

If gender goes from -1 for male to +1 for female, then the gender associated with man might be -1, for woman might be +1. And then eventually, learning these things maybe for king you get -0.95, for queen 0.97, and for apple and orange sort of genderless.

Another feature might be, well how royal are these things. And so the terms, man and woman are not really royal, so they might have feature values close to zero. Whereas king and queen are highly royal. And apple and orange are not really royal.

How about age? Man and woman doesn't connotes much about age. Maybe men and woman implies that they're adults, but maybe neither necessarily young nor old. So maybe values close to 0. Whereas kings and queens are always almost always adults. And apple and orange might be more neutral with respect to age.

And then, another feature for here, is this is a food? man is not a food, woman is not a food, neither are kings and queens, but apples and oranges are foods.

There can be many other features as well ranging from, what is the size of this? What is the cost? Is this something that is a live? Is this an action, or is this a noun, or is this a verb, or is it something else? And so on.

You can imagine coming up with many features.

![alt text](_assets/FeaturesOfWords.png)

For the sake of the illustration let's say, 300 different features, and what that does is, it allows you to take this list of numbers (list of numbers in column man), but this could be a list of 300 numbers, that then becomes a 300 dimensional vector for representing the word man.

I'm going to use the notation $e_{5391}$ to denote a representation like this.

Similarly, this vector, this 300 dimensional vector or 300 dimensional vector like this, I would denote $e_{9853}$ to denote a 300 dimensional vector we could use to represent the word woman. 

![alt text](_assets/NotationOfVector.png)

Similarly, for the other examples here.

Now, if you use this representation to represent the words orange and apple, then notice that the representations for orange and apple are now quite similar. Some of the features will differ because of the color of an orange, the color an apple, the taste, or some of the features would differ. But by a large, a lot of the features of apple and orange are actually the same, or take on very similar values. This increases the odds of the learning algorithm that has figured out that orange juice is a thing, to also quickly figure out that apple juice is a thing. So this allows it to generalize better across different words.

![alt text](_assets/FeaturizedRepresentation.png)

The features we'll end up learning, won't have a easy to interpret interpretation like that component one is gender, component two is royal, component three is age and so on. Exactly what they're representing will be a bit harder to figure out. But nonetheless, the featurized representations we will learn, will allow an algorithm to quickly figure out that apple and orange are more similar than say, king and orange or queen and orange.

### Visualizing word embeddings
If we're able to learn a 300 dimensional feature vector or 300 dimensional embedding for each words, one of the popular things to do is also to take this 300 dimensional data and embed it say, in a 2 dimensional space so that you can visualize them. 

One common algorithm for doing this is the t-SNE algorithm due to Laurens van der Maaten and Geoff Hinton.

If you look at one of these embeddings, one of these representations, you find that words like man and woman tend to get grouped together, king and queen tend to get grouped together, and these are the people which tends to get grouped together. Those are animals who can get grouped together. Fruits will tend to be close to each other. Numbers like one, two, three, four, will be close to each other. And then, maybe the animate objects as whole will also tend to be grouped together.

![alt text](_assets/WordsGrouping.png)

Maybe this gives you a sense that, word embeddings algorithms like this can learn similar features for concepts that feel like they should be more related, as visualized by that concept that seem to you and me like they should be more similar, end up getting mapped to a more similar feature vectors.

These representations we'll use these sort of featurized representations in maybe a 300 dimensional space, these are called embeddings.

The reason we call them embeddings is, you can think of a 300 dimensional space. And again, they can't draw out here in two dimensional space because it's a 3D one. And what you do is you take every words like orange, and have a three dimensional feature vector so that word orange gets embedded to a point in this 300 dimensional space. And the word apple, gets embedded to a different point in this 300 dimensional space.

And of course to visualize it, algorithms like t-SNE, map this to a much lower dimensional space, you can actually plot the 2D data and look at it. But that's what the term embedding comes from.

![alt text](_assets/WordEmbeddings.png)

Instead of sparse vectors, we use dense, learned vectors (e.g., 50–300 dimensions).
* Words with similar meanings end up with similar embeddings.
* Example relationships learned automatically:
  * king – man + woman ≈ queen
  * paris – france + italy ≈ rome

Word embeddings capture:
* Gender
* Royalty
* Type of food, places, sports
* Semantic and syntactic relationships

They allow generalization:
* If the model learns that “apple” is similar to “orange,” then:
  * If it understands “apple juice,” it will also understand “orange juice.”

This makes NLP models far more data-efficient and intelligent.
* Embeddings are high-dimensional (50–300 dims).
* Tools like t-SNE allow us to project them to 2D.
* Similar words form clusters, such as:
  * Countries together
  * Food together
  * Verbs together
  * Masculine vs feminine forms

This shows embeddings capture structure found in natural language.

## Using Word Embeddings
### Named entity recognition example

![alt text](_assets/NamedEntityRecog.png)

Continuing with the named entity recognition example, if you're trying to detect people's names.

Given a sentence like "Sally Johnson is an orange farmer", hopefully, you'll figure out that Sally Johnson is a person's name, hence, the outputs 1 like that.

And one way to be sure that Sally Johnson has to be a person, rather than say the name of the corporation is that you know orange farmer is a person.

So previously, we had talked about one hot representations to represent these words, $x^{<1>}$, $x^{<2>}$, and so on. 

But if you can now use the featurized representations, the embedding vectors that we talked about in the last video. Then after having trained a model that uses word embeddings as the inputs, if you now see a new input, "Robert Lin is an apple farmer". Knowing that orange and apple are very similar will make it easier for your learning algorithm to generalize to figure out that Robert Lin is also a human, is also a person's name.

One of the most interesting cases will be, what if in your test set you see not "Robert Lin is an apple farmer", but you see much less common words? What if you see "Robert Lin is a durian cultivator"?

A durian is a rare type of fruit, popular in Singapore and a few other countries.

But if you have a small label training set for the named entity recognition task, you might not even have seen the word durian or seen the word cultivator in your training set. But if you have learned a word embedding that tells you that durian is a fruit, so it's like an orange, and a cultivator, someone that cultivates is like a farmer, then you might still be generalize from having seen an orange farmer in your training set to knowing that a durian cultivator is also probably a person.

One of the reasons that word embeddings will be able to do this is the algorithms to learning word embeddings can examine very large text corpuses, maybe found off the Internet. So you can examine very large data sets, maybe a billion words, maybe even up to 100 billion words would be quite reasonable. So very large training sets of just unlabeled text. And by examining tons of unlabeled text, which you can download more or less for free, you can figure out that orange and durian are similar. And farmer and cultivator are similar, and therefore, learn embeddings, that groups them together.

Now having discovered that orange and durian are both fruits by reading massive amounts of Internet text, what you can do is then take this word embedding and apply it to your named entity recognition task, for which you might have a much smaller training set, maybe just 100,000 words in your training set, or even much smaller.

This allows you to carry out transfer learning, where you take information you've learned from huge amounts of unlabeled text that you can suck down essentially for free off the Internet to figure out that orange, apple, and durian are fruits. And then transfer that knowledge to a task, such as named entity recognition, for which you may have a relatively small labeled training set.

For simplicity, l drew this for it only as a unidirectional RNN. If you actually want to carry out the named entity recognition task, you should, of course, use a bidirectional RNN rather than a simpler one I've drawn here.

![alt text](_assets/NamedEntityExample.png)

### Transfer learning and word embeddings
To summarize, this is how you can carry out transfer learning using word embeddings.

1. Step 1 is to learn word embeddings from a large text corpus, a very large text corpus ((1-100B words)) or you can also download pre-trained word embeddings online.
2. Transfer embedding to new task with smaller training set.
(say, 100k words). One nice thing also about this is you can now use relatively lower dimensional feature vectors. So rather than using a 10,000 dimensional one-hot vector, you can now instead use maybe a 300 dimensional dense vector. Although the one-hot vector is fast and the 300 dimensional vector that you might learn for your embedding will be a dense vector.
3. Finally, as you train your model on your new task, on your named entity recognition task with a smaller label data set, one thing you can optionally do is to continue to fine tune, continue to adjust the word embeddings with the new data. In practice, you would do this only if this task 2 has a pretty big data set. If your label data set for step 2 is quite small, then usually, I would not bother to continue to fine tune the word embeddings.

So word embeddings tend to make the biggest difference when the task you're trying to carry out has a relatively smaller training set. So it has been useful for many NLP tasks.

It has been useful for named entity recognition, for text summarization, for co-reference resolution, for parsing. These are all maybe pretty standard NLP tasks.

It has been less useful for language modeling, machine translation, especially if you're accessing a language modeling or machine translation task for which you have a lot of data just dedicated to that task.

So as seen in other transfer learning settings, if you're trying to transfer from some task A to some task B, the process of transfer learning is just most useful when you happen to have a ton of data for A and a relatively smaller data set for B.

And so that's true for a lot of NLP tasks, and just less true for some language modeling and machine translation settings.

![alt text](_assets/TransferLearningWordEmbeddings.png)

### Relation to face encoding
Finally, word embeddings has a interesting relationship to the face encoding ideas that you learned about in the previous course, if you took the convolutional neural networks course.

So you will remember that for face recognition, we train this Siamese network architecture that would learn, say, a 128 dimensional representation for different faces. And then you can compare these encodings in order to figure out if these two pictures are of the same face. 

![alt text](_assets/SiameseNetwork.png)

The words encoding and embedding mean fairly similar things.

So in the face recognition literature, people also use the term encoding to refer to these vectors, $f(x^{(i)})$ and $f(x^{(j)})$. 

One difference between the face recognition literature and what we do in word embeddings is that, for face recognition, you wanted to train a neural network that can take as input any face picture, even a picture you've never seen before, and have a neural network compute an encoding for that new picture

Whereas what we'll do for learning word embeddings is that we'll have a fixed vocabulary of, say, 10,000 words. And we'll learn a vector $e_1$ through, say, $e_{10,000}$ that just learns a fixed encoding or learns a fixed embedding for each of the words in our vocabulary.

So that's one difference between the set of ideas you saw for face recognition versus what the algorithms we'll discuss in the next few videos.

But the terms encoding and embedding are used somewhat interchangeably.

So the difference I just described is not represented by the difference in terminologies. It's just a difference in how we need to use these algorithms in face recognition, where there's unlimited sea of pictures you could see in the future. Versus natural language processing, where there might be just a fixed vocabulary, and everything else like that we'll just declare as an unknown word.

So in this video, you saw how using word embeddings allows you to implement this type of transfer learning. And how, by replacing the one-hot vectors we're using previously with the embedding vectors, you can allow your algorithms to generalize much better, or you can learn from much less label data.

![alt text](_assets/RelationToFaceEncoding.png)

Word embeddings are a way to represent words as dense vectors (lists of numbers) so that similar words have similar vectors.

Instead of:
* “apple” → [0, 0, 0, 1, 0, 0 …] (one-hot)
* “orange” → [0, 1, 0, 0, 0, …]

Word embeddings represent words more meaningfully:
* “apple” → [0.82, 0.13, 0.55, …]
* “orange” → [0.80, 0.11, 0.52, …]

The numbers capture meaning, not just identity.

If the model learns that “orange” is a fruit, then the embedding places “durian” near it, allowing the model to generalize even if it has never seen “durian” during training.

Word embeddings let models generalize better.

Example: Named Entity Recognition (NER)

NER = Identify names of people, locations, companies in text.

Without embeddings:

If you never saw the word “Durian Corp” during training, the model has no clue it’s likely a company.

With embeddings:
* If “Durian” is close to “Orange” and “Apple” in vector space,
* and “Orange Corp” or “Apple Inc.” are often companies,
* the model can infer that “Durian Corp” might also be a company.

Embeddings bring semantic knowledge into the task, even if the training dataset for that task is very small.

Using knowledge learned from a large dataset to help with a smaller, labeled dataset.

In NLP:
* Train word embeddings on billions of unlabeled sentences (cheap to collect).
* Use these pre-trained embeddings for your specific task (with small labeled data).

Most NLP tasks don't have tons of labeled data, so transfer learning makes them possible.

Imagine building an NER system:
* You only have 10,000 labeled sentences → not much!
* But you load embeddings (Word2Vec, GloVe) trained on 1 billion sentences → you gain a huge amount of word knowledge instantly.

Imagine you're training a model to detect names of fruits in sentences, but your labeled dataset does not include “durian.”

Because embeddings learned from billions of sentences place:
* “durian” near “orange”
* and “orange” near “apple”
* and the model knows “apple” is a fruit from training

The model can guess:

-> “durian” is probably also a fruit.

This is the power of word embeddings + transfer learning.

Very Helpful For:
* Tasks where labeled data is limited:
* Named Entity Recognition (NER)
* Document classification
* Sentiment analysis
* Text summarization
* Co-reference resolution (e.g., knowing “John” = “he”)

Why?
-> They transfer knowledge from massive text corpora into your smaller task.

Less Helpful For:

Tasks where you can train a giant model on giant datasets:
* Machine translation (e.g., Google Translate)
* Language modeling (predict next word)

Why?
-> These tasks already train on enormous data, so pre-trained embeddings add little benefit.

In face recognition (from earlier in the specialization), the model also turns faces into vectors:
* Your face → [0.41, 0.12, 0.95, …]
* My face → [0.39, 0.11, 0.97, …]

These vectors encode the identity of the person, based on features like:
* Shape of eyes
* Nose
* Jawline
* Skin tone
* Etc.

Then:
* If two vectors are close → same person
* If they are far apart → different people

Just like similar words → embedding vectors close

similar faces → face encoding vectors close

The big idea:

Embeddings = a general technique in deep learning

You can represent anything with a learned vector:
* A word
* A face
* A product in a store
* A user in a recommendation system
* An image
* A sound clip

Once a neural network learns these embeddings, simple math (distances, comparisons) becomes possible.

Learn similarities
* Word: “apple” is like “orange”
* Face: My face is similar to yesterday’s picture of me

Learn differences
* Word: “apple” ≠ “bank”
* Face: My face ≠ your face

Work well with small amounts of data
* Word embeddings transfer knowledge from billions of text sentences
* Face encodings transfer knowledge from tons of face images

Deep learning often works by learning to represent things as vectors first.
Then those learned vectors make the downstream task easier.

## Properties of Word Embeddings
One of the most fascinating properties of word embeddings is that they can also help with analogy reasoning. And while reasonable analogies may not be by itself the most important NLP application, they might also help convey a sense of what these word embeddings are doing, what these word embeddings can do.

### Analogies
Here are the featurized representations of a set of words that you might hope a word embedding could capture. 

![alt text](_assets/FeaturizedRepresentationSetOfWords.png)

Let's say I pose a question, man is to woman as king is to what? 

Many of you will say, man is to woman as king is to queen. But is it possible to have an algorithm figure this out automatically? 

Let's say that you're using this four dimensional vector to represent man. So this will be your $e_{5391}$, although just for this video, let me call this $e_{man}$.

And let's say that's the embedding vector for woman, so I'm going to call that $e_{woman}$, and similarly for king and queen.

And for this example, I'm just going to assume you're using 4 dimensional embeddings, rather than anywhere from 50 to 1,000 dimensional, which would be more difficult.

One interesting property of these vectors is that if you take the vector, $e_{man}$, and subtract the vector $e_{woman}$, then, you end up with roughly [-2 0 0 0].

And similarly if you take e king minus e queen, then that's approximately the same thing. That's about [-2 0 0 0].

![alt text](_assets/AnalogiesExample.png)

What this is capturing is that the main difference between man and woman is the gender.

And the main difference between king and queen, as represented by these vectors, is also the gender.

Which is why the difference $e_{man}$ - $e_{woman}$, and the difference $e_{king}$ - $e_{queen}$, are about the same.

One way to carry out this analogy reasoning is, if the algorithm is asked, man is to woman as king is to what?

What it can do is compute $e_{man}$ - $e_{woman}$, and try to find a vector, try to find a word so that $e_{man}$ - $e_{woman}$ is close to $e_{lomg}$ - e of that new word. It turns out that when queen is the word plugged in here, then the left hand side is close to the the right hand side.

![alt text](_assets/Analogies.png)

These ideas were first pointed out by Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. And it's been one of the most remarkable and surprisingly influential results about word embeddings. And I think has helped the whole community get better intuitions about what word embeddings are doing.

### Analogies using word vectors
Let's formalize how you can turn this into an algorithm.

In pictures, the word embeddings live in maybe a 300 dimensional space. And so the word "man" is represented as a point in the space, and the word "woman" is represented as a point in the space. Same for "king" and "queen".

![alt text](_assets/WordEmbeddingInPicture.png)

What we pointed out really on the last slide is that the vector difference between man and woman is very similar to the vector difference between king and queen.

This arrow I just drew is really the vector that represents a difference in gender.

![alt text](_assets/VectorDifference.png)

Remember, these are points we're plotting in a 300 dimensional space.

In order to carry out this kind of analogical reasoning to figure out, man is to woman is king is to what, what you can do is try to find the word "w", so that, this equation holds true

$e_{man} - e_{woman} \approx e_{king} - e_?$

So you want to find the word "w" then finding the word that maximizes the similarity between $e_w$ compared to $e_{king} - e_{man} + e_{woman}$.

Find word w: ${argmax}_w$ sim($e_w$, $e_{king} - e_{man} + e_{woman}$)

What I did is, I took this $e_?$, and replaced that with $e_w$, and then brought $e_w$ to just one side of the equation. And then the other three terms to the right hand side of this equation.

We have some appropriate similarity function for measuring how similar is the embedding of some word "w" to this quantity of the right. Then finding the word that maximizes the similarity should hopefully let you pick out the word "queen".

The remarkable thing is, this actually works.

If you learn a set of word embeddings and find a word w that maximizes this type of similarity, you can actually get the exact right answer. Depending on the details of the task, but if you look at research papers, it's not uncommon for research papers to report anywhere from, say, 30% to 75% accuracy on analogy using tasks like these. Where you count an anology attempt as correct only if it guesses the exact word right. So only if, in this case, it picks out the word queen.

Previously, we talked about using algorithms like t-SAE to visualize words.

What t-SAE does is, it takes 300-D data, and it maps it in a very non-linear way to a 2D space. The mapping that t-SAE learns, this is a very complicated and very non-linear mapping. After the t-SAE mapping, you should not expect these types of parallelogram relationships, like the one we saw on the left, to hold true. And it's really in this original 300 dimensional space that you can more reliably count on these types of parallelogram relationships in analogy pairs to hold true. And it may hold true after a mapping through t-SAE, but in most cases, because of t-SAE's non-linear mapping, you should not count on that. And many of the parallelogram analogy relationships will be broken by t-SAE. 

![alt text](_assets/AnalogiesUsingWordVectors.png)

### Cosine similarity
sim($e_w$, $e_{king} - e_{man} + e_{woman}$)

In cosine similarity, you define the similarity between two vectors u and v as u transpose v divided by the lengths by the Euclidean lengths.

$sim(u,v) = {{u^Tv} \over {||u||_2 ||v||_2}}$

Ignoring the denominator for now, this is basically the inner product between u and v.

If u and v are very similar, their inner product will tend to be large.

This is called cosine similarity because this is actually the cosine of the angle between the two vectors, u and v. That's the angle phi, so this formula is actually the cosine of the angle between them.

You remember from calculus that if this phi, then the cosine of phi looks like this.

![alt text](_assets/cosinPhi.png)

If the angle between them is 0, then the cosine similarity is equal to 1. And if their angle is 90 degrees, the cosine similarity is 0. And then if they're 180 degrees, or pointing in completely opposite directions, it ends up being -1.

That's where the term cosine similarity comes from, and it works quite well for these analogy reasoning tasks.

If you want, you can also use square distance or Euclidian distance, $||u-v||^2$. Technically, this would be a measure of dissimilarity rather than a measure of similarity. So we need to take the negative of this, and this will work okay as well. Although I see cosine similarity being used a bit more often.

The main difference between these is how it normalizes the lengths of the vectors u and v.

One of the remarkable results about word embeddings is the generality of analogy relationships they can learn.

For example, it can learn that man is to woman as boy is to girl, because the vector difference between man and woman, similar to king and queen and boy and girl, is primarily just the gender. It can learn that Ottawa, which is the capital of Canada, that Ottawa is to Canada as Nairobi is to Kenya. So that's the city capital is to the name of the country. It can learn that big is to bigger as tall is to taller, and it can learn things like that. Yen is to Japan, since yen is the currency of Japan, as ruble is to Russia. 

All of these things can be learned just by running a word embedding learning algorithm on the large text corpus, it can spot all of these patterns by itself.

![alt text](_assets/CosineSimilarity.png)

So in this video, you saw how word embeddings can be used for analogy reasoning. And while you might not be trying to build an analogy reasoning system yourself as an application, this I hope conveys some intuition about the types of feature-like representations that these representations can learn.

You also saw how cosine similarity can be a way to measure the similarity between two different word embeddings.

### Summary
#### Property #1 — Word Embeddings Capture Meaningful Similarity

In embeddings:
* Words with similar meaning → have similar vectors
* Words with different meaning → have very different vectors

Example

“apple” and “orange” both appear near each other because:
* Both are fruits
* Both appear in similar sentences (buy apple / buy orange)

Meanwhile:
* “apple” and “car” are far apart because they appear in different contexts.

Your model can generalize better.

If your training data includes:
* “apple is a fruit”
* “orange is a fruit”

Then embeddings help the model guess:

“pear” is also likely a fruit even if it never saw the exact sentence during training.

#### Property #2 — Word Embeddings Capture Analogies
Embeddings don’t just understand similarity—they also understand relationships.

Famous examples
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

It means embeddings understand:
* Gender relationships
* Plural relationships
* Verb tense relationships
* Country–capital relationships

Examples:
```
Italy – Rome ≈ France – Paris  
Run – Running ≈ Swim – Swimming  
```

What does this mean in simple words?

Word embeddings organize words in such a way that the same type of relationship always points in the same direction in vector space.

For example:
* Adding the “plural direction” to “car” gets “cars”
* Adding the “past-tense direction” to “swim” gets “swam”

This shows that embeddings learn patterns of language, not memorization.

They let algorithms handle words they’ve never seen before: If embeddings place “durian” near “orange,” your classifier can guess that “durian” is also likely a fruit even without labeled training examples.

They reduce data needed for NLP tasks: Because embeddings bring “world knowledge” from big text corpora.

They make downstream tasks easier" Sentiment analysis, named entity recognition, text classification all benefit.

They let you use simple math to reason about words" Instead of thinking hard about language rules, the model just compares vectors.

Andrew shows plots like this (but in higher dimensions):
* Gender direction
* Country–capital direction
* Verb tense direction

Think of embeddings as forming geometric patterns:

```
king    → queen  
man     → woman  
Rome    → Italy  
Paris   → France  
walking → walked  
swimming → swam
```

All these relationships become parallel vector shifts, which is why vector math works.

## Embedding Matrix
When you implement an algorithm to learn a word embedding, what you end up learning is an embedding matrix.

Let's say, as usual we're using our 10,000-word vocabulary. So, the vocabulary has A, Aaron, Orange, Zulu, maybe also unknown word as a token.

What we're going to do is learn embedding matrix E, which is going to be a 300 dimensional by 10,000 dimensional matrix, if you have 10,000 words vocabulary or maybe 10,001 is unknown word token,there's one extra token.

The columns of this matrix would be the different embeddings for the 10,000 different words you have in your vocabulary.

Orange was word number 6257 in our vocabulary of 10,000 words.

One piece of notation we'll use is that $O_{6257}$ was the one-hot vector with 0's everywhere and a 1 in position 6257.

This will be a 10,000-dimensional vector with a 1 in just one position.

![alt text](_assets/VocabularyMatrixExample.png)

This isn't quite a drawn scale. This should be as tall as the embedding matrix on the left is wide.

If the embedding matrix is called capital E then notice that if you take E and multiply it by this one-hot vector by $O_{6257}$, then this will be a 300-dimensional vector. So, E is 300 by 10,000 and O is 10,000 by 1.

The product will be 300 by 1, so with 300-dimensional vector.

Notice that to compute the first element of this vector, of this 300-dimensional vector, what you do is you will multiply the first row of the matrix E with $O_{6257}$. But all of these elements are zero except for element 6257 and so you end up with the first element as whatever is that elements up there, under the Orange column.

![alt text](_assets/EmbeddingMatrix1.png)

Then, for the second element of this 300-dimensional vector we're computing, you would take the vector $O_{6257}$ and multiply it by the second row with the matrix E.

Again, you end up with this and so on as you go down the rest of this column.

![alt text](_assets/ProductOfEAndO.png)

That's why the embedding matrix E times this one-hot vector winds up selecting out this 300-dimensional column corresponding to the word Orange.

This is going to be equal to $E_{6257}$ which is the notation we're going to use to represent the embedding vector that 300 by 1 dimensional vector for the word Orange. 

![alt text](_assets/EmbeddingMatrix2.png)

## Learning Word Embeddings









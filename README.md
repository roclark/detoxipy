# Detoxipy
Detoxipy is a simple text classifier used to identify toxic comments with the
aim that it can be used by developers to quickly flag potentially harmful or
hateful comments in any applications which involve textual communication. The
project currently contains two classifiers, the **Binary Classifier** and the
**Multilabel Classifier** which solve the same task, but provide slightly
different outputs.

## Classifiers

### Binary Classifier
The Binary Classifier is a simple classifier which outputs a `1` if a comment is
toxic, and a `0` if not. For example, the non-toxic comment of "Have a nice
day!" will output a `0`. The Binary Classifier is the most efficient algorithm
which can determine whether or not a comment is toxic in approximately 10
milliseconds.

### Multilabel Classifier
The Multilabel Classifier is slightly more complex as comments can fall into
several of up to seven unique categories. These categories are the following:
  * Toxic - A comment which can generally be described as harmful
  * Severe Toxic - A comment which is very harmful and will often also fall into
one or more of the other categories
  * Obscene - These comments are typically viewed as overtly sexual, and/or
morally offensive
  * Threat - Any comment which indicates intentions to harm, typically mentally,
physically, or emotionally, another person
  * Insult - These comments include generally hateful comments towards another
person
  * Identity Hate - These comments are typically insults towards someones
gender, race, sexual orientation, or others
  * Non-Toxic - Any comment that doesn't fall into the previous categories is
determined to be non-toxic

Any comment can fall into one or multiple of the first six comments, such as
being "toxic", "severe toxic", and "obscene". The only exception comes when a
comment is determined non-toxic, where it can only fall into that single
category. While the Multilabel Classifier provides greater details on what
classification of toxicity a comment falls into, it takes longer to generate a
response compared to the Binary Classifier while still being responsive by
averaging approximately 90 milliseconds to complete.

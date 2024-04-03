### RL4LLM - Spring 2024

This repository contains the experiment code for the Fairness and Emotion Understanding Experiments, and a description of the methods. 

Experiments and Results: 

The primary experiments are run on the [EmoSet Dataset](https://vcc.tech/EmoSet). This dataset contains in-the-wild images, and the ground truth emotion ratings are primarily based on emotions expected to be evoked in the viewer. The experiments are run using the LLaVA model. The form of prompting uses direct close questioning, providing the model with a list of emotions from which to choose. It stands out from existing work in the sense that all previous works provide LLMs or VLMs with some kind of additional context to generate emotional context, whereas these experiments test them in a discriminative approach. 

Update on experimental results: 
* With simple multimodal prompts (just the Image and the list of emotions to choose from), weighted F1 scores achieved are poor (~0.11), whereas the precision is very high (~0.99), and the recall achieved is again low (~0.09). This means that the model predicts all images and prompts to belong to a single emotion category.
* With multimodal prompts that also ask for an explanation, the F1, precision, and recall remain the same. However, the explanations generated provide additional insight into where the model goes wrong. It confuses negative emotions like "Anger" and "Sadness", and although it can locate the key objects in the image provided, it deduces an incorrect emotion from the key objects, pointing to an affective gap. Consider the following examples below:

  ***Ground truth label = anger
     Predicted label =  sadness
     Explanation generated = Sadness. The image shows a group of people holding signs, which could indicate a protest or a demonstration.***

In some cases, although it recognizes the key object correctly in the image, it misreads the expression or the context and deduces the incorrect emotion. 

  ***Ground truth label = anger
     Predicted label =  amusement
     Explanation generated = Amusement. The bird has a wide open mouth and is making a funny face.***

Remediation, Redesign, Revise: 

Two changes had to be made for logistical purposes:
1. Using LLaVA directly, without AutoGen, as the server I used ran into network errors when hosting the model worker and controller.
2. Adding new prompts that encourage providing explanations for predictions made, and do chain-of-thought reasoning.

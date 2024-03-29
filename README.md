### RL4LLM - Spring 2024

This repository contains the experiment code for the Fairness and Emotion Understanding Experiments, and a description of the methods. 

Experiments and Results: 

The primary experiments are run on the [EmoSet Dataset](https://vcc.tech/EmoSet). This dataset contains in-the-wild images, and the ground truth emotion ratings are primarily based on emotions expected to be evoked in the viewer. The experiments are run using the LLaVA model. The form of prompting uses direct close questioning, providing the model with a list of emotions from which to choose. It stands out from existing work in the sense that all previous works provide LLMs or VLMs with some kind of additional context to generate emotional context, whereas these experiments test them in a discriminative approach. 

Remediation, Redesign, Revise: 

Two changes had to be made for logistical purposes:
1. Using LLaVA directly, without AutoGen, as the server I used ran into network errors when hosting the model worker and controller.
2. Adding new prompts that encourage providing explanations for predictions made, and do chain-of-thought reasoning.

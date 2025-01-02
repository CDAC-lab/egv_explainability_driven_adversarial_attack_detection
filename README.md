# Explainable Artificial Intelligence with Integrated Gradients for the Detection of Adversarial Attacks on Text Classifiers

ext classifiers are Artificial Intelligence (AI) models used to classify new documents or text
vectors into pre-defined classes. They are typically built using supervised learning algorithms and
labelled datasets. Text classifiers produce a pre-defined class as an output, which also makes them
susceptible to adversarial attacks. Text classifiers with high accuracy that are trained using complex
deep learning algorithms are equally susceptible to adversarial examples, due to subtle differences
that are indiscernible to human experts. Recent work in this space is mostly focused on improving
adversarial robustness and adversarial example detection, instead of detecting adversarial attacks.
In this work, we propose a novel approach, Explainable AI with Integrated Gradients (IGs) for the
detection of adversarial attacks on text classifiers. This approach uses IGs to unpack model behaviour
and identify terms that positively and negatively influence the target prediction. Instead of random
substitution of words in the input, we select the top k words with the greatest positive and negative
influence as substitute candidates using attribution scores obtained from IG to generate k samples
of transformed inputs by replacing them with synonyms. This approach does not require changes
to the model architecture or the training algorithm. The approach was empirically evaluated on
three benchmark datasets IMDB, SST-2 and AG News. Our approach outperforms baseline models
on word substitution rate, detection accuracy and F1 scores while maintaining equivalent detection
performance against adversarial attacks.

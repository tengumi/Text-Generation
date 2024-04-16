# Text Generation

Text Generation is a type of Language Modelling problem. Language Modelling is the core problem for a number of of natural language processing tasks such as speech to text, conversational system, and text summarization. A trained language model learns the likelihood of occurrence of a word based on the previous sequence of words used in the text. Language models can be operated at character level, n-gram level, sentence level or even paragraph level. In this notebook, I will explain how to create a language model for generating natural language text by implement and training state-of-the-art Long short-term memory(LSTM).

## About The Project

Welcome, in this project the goal is to make a text generator using an LSTM model trained on the first Harry Potter book and then improve the quality of the model by using a transformer. Text generation is a very interesting task in NLP. Let's see what we have to interact with!

## LSTM

[Reference](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.

### Brief structure

![LSTM_core](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

<div style="text-align: center;"><strong>The repeating module in an LSTM contains four interacting layers.</strong></div><br>

![vectores](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png)

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

| Blocks                                                                                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" width="4000"> | The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ht−1 and xt, and outputs a number between 0 and 1 for each number in the cell state Ct−1. A 1 represents “completely keep this” while a 0 represents “completely get rid of this.”                                             |
| <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" width="4000"> | The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C~t, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.                                         |
| <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png" width="4000"> | It’s now time to update the old cell state, Ct−1, into the new cell state Ct. The previous steps already decided what to do, we just need to actually do it. We multiply the old state by ft, forgetting the things we decided to forget earlier. Then we add it∗C~t. This is the new candidate values, scaled by how much we decided to update each state value.                                                           |
| <img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" width="4000"> | Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to. |

### Types of problems that can be solved with LSTMs

- Time Series Prediction
- Natural Language Processing
- Tone analysis:
- Speech Recognition:
- Time series analysis
- Sequence prediction

## Decoder model (GPT-2)

[Reference](https://huggingface.co/openai-community/gpt2)

GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences.

<div align="center">

![GPT-2](https://av-eks-lekhak.s3.amazonaws.com/media/__sized__/article_images/image_9QB0Fmx-thumbnail_webp-600x300.webp)

</div>

# The main components of GPT-2:

    Decoder Transformer: This is the main part of the model that utilizes the Transformer architecture. Transformer is a model that uses self-attention mechanism to analyze input data and generate output data. GPT-2 uses decoder which is used to generate text.

    Self-Attention: It is a mechanism that allows the model to analyze input data and generate output data. In GPT-2 self-attention is used in each layer of decoder.

    Multi-Head Attention: It is a mechanism that allows the model to analyze input data and generate output data. In GPT-2 multi-head attention is used in each layer of decoder.

    Feed Forward Neural Network: It is a simple neural network which is used in each layer of decoder.

    Positional Encoding: It is a mechanism that adds positional information to the input data.

    Layer Normalization: It is a mechanism that normalizes the output data of each layer.

    Dropout: This is a mechanism that is used to regularize the model and prevent overtraining.

    Softmax: This is an activation function that is used to generate probabilities for each possible output token.

<p align="center">
  <img src="https://res.cloudinary.com/edlitera/image/upload/c_fill,f_auto/v1680629118/blog/gz5ccspg3yvq4eo6xhrr" alt="decoder" style="width:25%; height:auto;">
</p>

In general, GPT-2 uses Transformer to generate text. It uses self-attention to analyze input data and generate output data. Each layer of decoder uses multi-head attention, feed forward neural network, layer normalization and dropout. Positional information is added using positional encoding. Softmax is used to generate probabilities for each possible output token.

## Summarize

As expected, GPT-2 performed better than LSTM, but both models can be used to solve this problem, but a lot of time and computational power will have to be spent to train LSTM, so it is better to use out-of-the-box solutions such as Transformers for convenience.

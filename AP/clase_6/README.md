# Clase 5 - Recurrent Neural Network

## [Teoria](teoria/RNN.E2.pptx) (powerpoint):
- Red recurrente básica, ecuaciones.
- Unfolding de una RNN.
- RNN en Pytorch y dimensiones. ([práctica 1](jupyter_notebooks/RNN_teoria.ipynb)).
- Back propagation through time (BPTT)
- Implementación: predicción de serie temporal. ([práctica 2](jupyter_notebooks/RNN_signal_wandb.ipynb)).
- Encoder - Decoder. ([ver colab](jupyter_notebooks/RNN_enc_dec.ipynb)).
- Attention.

## Información adicional
- [Back Propagation Through Time - BPTT](teoria/BPTT_desarrollo.pdf)
- [Presentación con Attention](teoria/cs224n-2021-lecture07-nmt.pdf)

## Notebooks:
- [RNN_teoria](jupyter_notebooks/RNN_teoria.ipynb) (ejercicios de dimensiones en RNN).
- [RNN_teoria_resolución](jupyter_notebooks/RNN_teoria_resolución.ipynb) (resolución de los ejercicios de dimensiones en RNN).
- [RNN_signal_wandb](jupyter_notebooks/RNN_signal_wandb.ipynb) (implementación de RNN para predecir una serie temporal).
- [RNN_enc_dec](jupyter_notebooks/RNN_enc_dec.ipynb) (implementación de un ENC-DEC para la serie temporal y predición de 10 valores a futuro).


## Ejercitción para luego de la clase:
Generar distintas arquitecturas de RNN que mejoren la performance del MLP. Compare la cantidad de parámetros con el MLP y prueve entre la distintas RNN diversos hiperparámetros.


## Bibliografía:

- deep learning book
https://www.deeplearningbook.org/contents/rnn.html

- BPTT con buena explicación de los vanishing y exploding gradients:
https://mmuratarat.github.io/2019-02-07/bptt-of-rnn

- Demostración de derivada de Loss function w.r.t. softmax:
https://mmuratarat.github.io/2019-02-10/derivative-of-softmax-loss

- Attention mechanism con "animaciones" interesantes:
https://distill.pub/2016/augmented-rnns/

## Implementaciones end-to-end interesantes (usan redes mas avanzadas):

- LSTM sentiment analysis (largo de correr...)
https://github.com/gabrielloye/LSTM_Sentiment-Analysis

- LSTM para estimación de demanda de energía:
https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b



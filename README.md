# VoteNet - NPM3D Course Project, Master MVA
 
Project for the course "Nuages de Points et Modélisation 3D" of the Master MVA (2020-2021).

The goal of the project was to study the method described in the paper "Deep Hough Voting for 3D Object Detection in Point Clouds", test an implementation of the method, and reproduce some results.

Most of the code comes from the original github repository: https://github.com/facebookresearch/votenet. The folders `models` and `losses` contain my own rewriting of the model's architecture and loss functions, with reorganized code and some additional comments. I did this rewriting to better understand the inner mechanisms of the method, and the behaviour of this implementation is similar to the official one. The training scripts `main_train.py` and `main_eval.py` are also adapted from the original code.

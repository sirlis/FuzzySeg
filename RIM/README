You can run unsupervised clustering on an NxN kernel matrix K with:

params.max_class = 25; % maximum number of clusters
params.algo = 'kernel';  % may be 'kernel' or 'linear'
params.lambda = 1; % regularization parameter
model = RIM(K,[],[],params);

model.alphas contains the kernel weights and model.bs contains the biases. 

There are a number of options such as initializing with k-means (Piotr Dollar's kmeans2 may be downloaded in his toolbox online), 
semi-supervised learning, or training a linear model.  Hopefully they're self explanatory from examining RIM.m, but please let me know 
if you have any problems/questions, or if I have forgotten anything.

Ryan Gomes
gomes@vision.caltech.edu
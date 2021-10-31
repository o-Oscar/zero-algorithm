# zero-algorithm

This repo contains a bare-bone implementation of the alpha-zero algorithm for single-player games.
We also implemented several algorithms that resemble alphazero with some component missing (action network for example).

To benchmark the performance of those algorithms with a few different hyper-parameters, we use the game of the hanoi towers.
Since some algorithms use much more compute per epoch, we can not make a fair comparaison between algorithms with a fixed number of epoch. Thus, we rather use a fixed time budget to benchmark the algorithms against one another. We make the assumption that all of the algorithms are as poorly optimized. Thus, the performance ratio between two algorithms will stay the same with optimization. 


## Summary from Andrej Karpathy's [A recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/)
* Neural Networks fail silently, they are hard to unit test as the possible error surface is huge. A lot of times this happens due to misconfiguration (not using training mean for scaling, wrong learning rate, decay reates, clipping losses instead of gradients.
* **Inspect your data:** 
  * Check for duplicates, corruptions, data imbalance, biases.
  * Are local features enough or do we need a global context. What variation is spurious and can be preprocessed.
  * How noisy are labels?
  * Since neural networks are a compressed version of your data, if the network is giving some prediction that doesnt seem consistent with the data, something is wrong.
  * Visualize distribution of features and check outliers (source of bugs in prediction)
* **Evaluation:**
  * Pick a simple model like linear classifier or simple CNN and set some dumb baselines.
  * Fix random seed, dont data augment at this stage, verify initial loss for correctness, intialize weights well, train an input-independent baseline (inputs = zero), overfit one batch, verify decreasing training loss, visualize what exactly goes into your network (model input), use backprop to chart dependencies
* **Overfit:**
  * Compute a metric you can trust.
  * For the right model: Get a large enough model such that it can overfit and then regularize it appropriately.
  * Find the simplest architecture to solve your problem
  * Adam optimizer is safe
  * Plug in signals one by one instead of all at once to evaluate performance boost
  * Use your own learning rate decay schedules. Initially play around with a fixed LR
* **Regularize:**
  * Get more data. Otherwise use ensembles (dont ensemble more than 5 models)
  * Pretrain
  * Stick with supervised learning
  * Use smaller input dimensions
  * Smaller model size
  * Decrease the batch size
  * Add dropout (Use sparingly as drop out does not play nice with batch normalization)
  * weigh decay
  * early stopping
  * try a larger model (last resort)
  * Visualize network's first layer weights and ensure you get nice edges. If filters look like noise, somthing is wrong
* **Tune:**
  * Random grid search for hyperparameters
  * Bayesian hyperparameter optimization


## Questions

1. **When building a neural network, should you overfit or underfit it first?**</br>
**Ans**: Overfitting is likely to be worse than underfitting. The reason is that there is no real upper limit to the degradation of generalisation performance that can result from over-fitting, whereas there is for underfitting. But you should overfit to a sample and then try regularization techiques to improve it.
<hr/>

2.**Draw the graphs for sigmoid, tanh, ReLU, and leaky ReLU.**</br>
**Ans**: ![image](https://user-images.githubusercontent.com/29446732/214356683-43a69325-fffb-4198-9b8d-cfd603c72f29.png)
<hr/>

3.**Pros and cons of each activation function.**</br>
**Ans**: Read: https://medium.com/analytics-vidhya/comprehensive-synthesis-of-the-main-activation-functions-pros-and-cons-dab105fe4b3b 
<hr/>

4.**Is ReLU differentiable? What to do when it’s not differentiable?**</br>
**Ans**: ReLU(z) is differentiable in all of its domain except when z=0. Incase of z = 0, we simply use derivative as 0. [Ref](https://sebastianraschka.com/faq/docs/relu-derivative.html)
<hr/>

5.**Derive derivatives for sigmoid function when is a vector.** </br>
**Ans**: Derivative of sigmoid is f(y)(1-f(y)), where f(y) is the sigmoid. More on the derivation: https://hausetutorials.netlify.app/posts/2019-12-01-neural-networks-deriving-the-sigmoid-derivative/
<hr/>

6.**What’s the motivation for skip connection in neural works?** </br>
**Ans**: To sum up, the motivation behind this type of skip connections is that they have an uninterrupted gradient flow from the first layer to the last layer, which tackles the **vanishing gradient problem**. More reading: https://theaisummer.com/skip-connections/
<hr/>

7.**How do we know that gradients are exploding? How do we prevent it?**</br>
**Ans**: When large error gradients accumulate, exploding gradients occur, resulting in very large updates to neural network model weights during training. The weights’ values can also grow to the point where they overflow, resulting in NaN values. Found a lot in RNN architectures. Use Gradient cliiping to cut exploding gradients. Gradient clipping involves introducing a pre-determined gradient threshold and then scaling down gradient norms that exceed it to match the norm. In general, exploding gradients can be avoided by carefully configuring the network model, such as using a small learning rate, scaling the target variables, and using a standard loss function. More reading: https://analyticsindiamag.com/how-can-gradient-clipping-help-avoiding-the-exploding-gradient-problem/
<hr/>

8.**Why are RNNs especially susceptible to vanishing and exploding gradients?**</br>
**Ans**: In RNNs exploding gradients happen when trying to learn long-time dependencies, because retaining information for long time requires oscillator regimes and these are prone to exploding gradients [Ref](https://stats.stackexchange.com/questions/140537/why-do-rnns-have-a-tendency-to-suffer-from-vanishing-exploding-gradient)
<hr/>

9.**Weight normalization separates a weight vector’s norm from its gradient. How would it help with training?** </br>
**Ans**: Speeds up the convergence process by optimizing gradient descent. More on this: https://sh-tsang.medium.com/review-weight-normalization-a-simple-reparameterization-to-accelerate-training-of-deep-neural-c88fb0918c9b
<hr/>

10.**When training a large neural network, say a language model with a billion parameters, you evaluate your model on a validation set at the end of every epoch. You realize that your validation loss is often lower than your train loss. What might be happening?** </br>
**Ans**: 
11. What criteria would you use for early stopping?
12. Gradient descent vs SGD vs mini-batch SGD.
13. It’s a common practice to train deep learning models using epochs: we sample batches from data without replacement. Why would we use epochs instead of just sampling data with replacement?
14. Your model’ weights fluctuate a lot during training. How does that affect your model’s performance? What to do about it?
15. Draw a graph number of training epochs vs training error for when the learning rate is:

    too high
    too low
    acceptable.
16. What’s learning rate warmup? Why do we need it?
17. Compare batch norm and layer norm.
18.  Why is squared L2 norm sometimes preferred to L2 norm for regularizing neural networks?
19.  Some models use weight decay: after each gradient update, the weights are multiplied by a factor slightly less than 1. What is this useful for?
20.  It’s a common practice for the learning rate to be reduced throughout the training.

    [E] What’s the motivation?
    [M] What might be the exceptions?
21. What happens to your model training when you decrease the batch size to 1?
22. What happens when you use the entire training data in a batch?
23.  How should we adjust the learning rate as we increase or decrease the batch size?
24.  Why is Adagrad sometimes favored in problems with sparse gradients?
25.  What can you say about the ability to converge and generalize of Adam vs. SGD? What else can you say about the difference between these two optimizers? 
26.  With model parallelism, you might update your model weights using the gradients from each machine asynchronously or synchronously. What are the pros and cons of asynchronous SGD vs. synchronous SGD?

27. **Why are neural networks with hidden units not convex?** </br>
**Ans:** If we have two identical structure of neural networks and we do a permutation of the hidden units in a way that preserves the function being computed, we will get a new minima with a different set of weights. Therefore the function learnt by the neural network will not be convex as there's no global minima.

28. **Problems encountered while training neural networks and their diagnostics** </br><img width="623" alt="image" src="https://user-images.githubusercontent.com/29446732/214734985-aa9ed98a-1f09-4375-b0df-27e746a2746f.png">

  


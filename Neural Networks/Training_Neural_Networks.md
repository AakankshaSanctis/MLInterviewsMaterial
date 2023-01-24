
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
* Regularize
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
* Tune
  * Random grid search for hyperparameters
  * Bayesian hyperparameter optimization
  


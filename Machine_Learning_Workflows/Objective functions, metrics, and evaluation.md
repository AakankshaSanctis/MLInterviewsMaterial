1. **Convergence.**
1. [E] When we say an algorithm converges, what does convergence mean?
   1. If optimization is a process that generates candidate solutions, then convergence represents a stable point at the end of the process when no further changes or improvements are expected. For eg if the loss does not continue to change after a few iterations, we say that the loss has converged.
1. [E] How do we know when a model has converged?
   1. A machine learning model reaches convergence **when it achieves a state during training in which loss settles to within an error range around the final value**. In other words, a model converges when additional training will not improve the model.

\2. [E] **Draw the loss curves for overfitting and underfitting.**

![](Aspose.Words.4d37bd4e-95e5-4072-89be-7c3f27de2772.001.png)


**3. Bias-variance trade-off**

1. [E] What’s the bias-variance trade-off?
   1. In statistics and machine learning, the bias–variance tradeoff is **the property of a model that the variance of the parameter estimated across samples can be reduced by increasing the bias in the estimated parameters**.
   1. As the complexity of the model increases, the bias (test error when train data is infinite) of the model decreases but the variance of the model increases. Model becomes sensitive to small changes.
1. [M] How’s this tradeoff related to overfitting and underfitting?
   1. An overfit model has low bias and high variance while an underfit model has high bias and low variance
1. [M] How do you know that your model is high variance, low bias? What would you do in this case?
   1. If the model is complex, has more trainable parameters, then we know that the model may have higher variance, lower bias. Incase validation loss starts increasing and training loss continues to decrease, we know that we have an overfit model, which has high variance, low bias.
1. [M] How do you know that your model is low variance, high bias? What would you do in this case?
   1. If the model is simple, less trainable parameters, then the model has low variance, high bias. If the train error is significantly large, the model is underfit and hence has higher bias.

1. **Cross-validation.**
   1. [E] Explain different methods for cross-validation.
      1. <https://towardsdatascience.com/understanding-8-types-of-cross-validation-80c935a4976d>
      1. https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right
   1. [M] Why don’t we see more cross-validation in deep learning?
      1. Cross-validation in Deep Learning (DL) might be a little tricky because most of the CV techniques require training the model at least a couple of times. 
      1. In deep learning, you would normally tempt to avoid CV because of the cost associated with training k different models. Instead of doing k-Fold or other CV techniques, you might use a random subset of your training data as a hold-out for validation purposes.

1. **Train, valid, test splits.**

   1. [E] What’s wrong with training and testing a model on the same data?
      1. The problem of training and testing on the same dataset is that **you won't realize that your model is overfitting**, because the performance of your model on the test set is good. The purpose of testing on data that has not been seen during training is to allow you to properly evaluate whether overfitting is happening
   1. [E] Why do we need a validation set on top of a train set and a test set?
      1. Now the validation dataset is useful when it comes to hyper-parameter tuning and model selection. The validation examples included in this set will be used to find the optimal values for the hyper-parameters of the model under consideration.
      1. https://towardsdatascience.com/training-vs-testing-vs-validation-sets-a44bed52a0e1
   1. [M] Your model’s loss curves on the train, valid, and test sets look like this. What might have been the cause of this? What would you do?
      1. Overfitting, decrease the model parameters. Reduce the number of iterations
1. ![Problematic loss curves](Aspose.Words.4d37bd4e-95e5-4072-89be-7c3f27de2772.002.png)

1. [E] Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim?
   1. Ask whether the accuracy is on the train sample or on an unseen sample. Do multiple tests in real life data to check if the same accuracy is achieved. Make sure model is not overfit. In this case, it most likely looks like an imbalanced class situation with number of cancer patients being significantly lesser. Check other metrics such as F1 score .
1. **F1 score.**
   1. [E] What’s the benefit of F1 over the accuracy?
      1. Both of those metrics take class predictions as input so you will have to adjust the threshold regardless of which one you choose. Remember that the F1 score is **balancing precision and recall on the positive class** while accuracy looks at correctly classified observations both positive and negative
         https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
   1. [M] Can we still use F1 for a problem with more than two classes. How?
      1. Unlike binary classification, multi-class classification generates an F-1 score for each class separately.
      1. https://www.baeldung.com/cs/multi-class-f1-score
1. For logistic regression, why is log loss recommended over MSE (mean squared error)?
   1. One of the main reasons why MSE doesn't work with logistic regression is when the MSE loss function is plotted with respect to weights of the logistic regression model, **the curve obtained is not a convex curve which makes it very difficult to find the global minimum**.
   1. https://towardsdatascience.com/why-not-mse-as-a-loss-function-for-logistic-regression-589816b5e03c
1. [M] When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?
   1. https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d
1. [M] Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.
   1. https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81
1. [M] For classification tasks with more than two labels (e.g. MNIST with 10 labels), why is cross-entropy a better loss function than MSE?
   1. https://stats.stackexchange.com/questions/573944/why-is-cross-entropy-loss-better-than-mse-for-multi-class-classification
1. [E] Consider a language with an alphabet of 27 characters. What would be the maximal entropy of this language?
   1. p=1/27, calculate entropy. https://medium.com/geekculture/how-to-calculate-the-entropy-of-an-entire-language-d17135b01282
1. A lot of machine learning models aim to approximate probability distributions. Let’s say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?
   1. KL Divergence: https://dibyaghosh.com/blog/probability/kldivergence.html
1. MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)
   1. [E] How do MPE and MAP differ?
   1. [H] Give an example of when they would produce different results.
1. [E] Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than 10% from the actual price. Which metric would you use?

   ` `**Hint**: check out MAPE.

**In which situations do we choose precision over recall and vice-versa?** </br>
**Ans**: https://datascience.stackexchange.com/questions/30881/when-is-precision-more-important-over-recall



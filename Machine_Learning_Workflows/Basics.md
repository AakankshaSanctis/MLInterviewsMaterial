# Basics


1. **Explain supervised, unsupervised, weakly supervised, semi-supervised, reinforcement learning, self-supervised learning and active learning** </br>
* **Supervised learning**: Supervised learning (SL) is a machine learning paradigm for problems where the available data consists of labelled examples, meaning that each data point contains features (covariates) and an associated label. The goal of supervised learning algorithms is learning a function that maps feature vectors (inputs) to labels (output), based on example input-output pairs.  
* **Unsupervised learning**: Unsupervised learning is a type of algorithm that learns patterns from untagged data. Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data. Two types of algorithms are used: Clustering and Association.
* **Weakly supervised learning**: Weak supervision is a branch of machine learning where noisy, limited, or imprecise sources are used to provide supervision signal for labeling large amounts of training data in a supervised learning setting. This approach alleviates the burden of obtaining hand-labeled data sets, which can be costly or impractical. Eg. Global statistics, Weak classifiers etc.
* **Semi-supervised learning**: Semi-supervised learning is a learning problem that involves a small number of labeled examples and a large number of unlabeled examples. Learning problems of this type are challenging as neither supervised nor unsupervised learning algorithms are able to make effective use of the mixtures of labeled and untellable data. As such, specialized semis-supervised learning algorithms are required. The primary difference, though, is that semi-supervised learning propagates knowledge (“based on what is already labeled, label some more”) whereas weak supervision injects knowledge (“based on your knowledge, label some more”) [Some heuristics]
* **Active learning**: Active learning is the subset of machine learning in which a learning algorithm can query a user interactively to label data with the desired outputs. In active learning, the algorithm proactively selects the subset of examples to be labeled next from the pool of unlabeled data. The fundamental belief behind the active learner algorithm concept is that an ML algorithm could potentially reach a higher level of accuracy while using a smaller number of training labels if it were allowed to choose the data it wants to learn from. Three types of active learning are: Stream-based selective sampling, pool-based sampling and membership query synthesis. 
* **Reinforcement learning**: Reinforcement Learning(RL) is a type of machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences. Though both supervised and reinforcement learning use mapping between input and output, unlike supervised learning where the feedback provided to the agent is correct set of actions for performing a task, reinforcement learning uses rewards and punishments as signals for positive and negative behavior
* **Self-supervised learning**: Self-Supervised Learning (SSL) is a Machine Learning paradigm where a model, when fed with unstructured data as input, generates data labels automatically, which are further used in subsequent iterations as ground truths. The fundamental idea for self-supervised learning is to generate supervisory signals by making sense of the unlabeled data provided to it in an unsupervised fashion on the first iteration. Then, the model uses the high confidence data labels among those generated to train the model in the next iterations like any other supervised learning model via backpropagation. The only difference is, the data labels used as ground truths in every iteration are changed. 
<hr/>

2. **What is [emphirical risk minimization]**(https://towardsdatascience.com/learning-theory-empirical-risk-minimization-d3573f90ff77)? </br>
**Ans**: The difference between the emphirical error (on a sample S sampled from distribution D) and the true error. Ideally, we want the difference between these two errors to be minimum.</br></br>
    2.1 **What’s the risk in empirical risk minimization?** </br>
    **Ans**: The error is called the risk in emphirical risk minimization.</br></br>
    2.2 **Why is it emphirical?**</br>
    **Ans**: Because it is computed on a sample of the original distribution and not on the              entire distribution.</br></br>
    2.3 **How do we minimize that risk?** </br>
    **Ans**: Avoid overfitting, this will decrease the emphirical error but significantly                increase the true error. Using regularization (L2) will help in this case.                  Increasing the sample size will also decrease the emphirical and true error. 
    <hr/>
    
 3. **Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?** </br>
 **Ans**:  If two models have the same performance on the validation/testing dataset select the simpler model because it is more likely to generalize well. Occam’s Razor is really just an example of the bias-variance tradeoff in machine learning. When selecting a model to use for any problem, we want a model that is complex enough to avoid underfitting, and simple enough to avoid overfitting. [Ref](https://towardsdatascience.com/what-occams-razor-means-in-machine-learning-53f07effc97c)
 <hr/>
 
 4. **What are the conditions that allowed deep learning to gain popularity in the last decade?** </br>
 **Ans**: The availability of massive amounts of data for training computer systems. Availability of GPU cluster on cloud and distributed processes.
 <hr/>
 
 5. **If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?** </br>
 **Ans**: The main issue is that very wide, shallow networks are very good at memorization, but not so good at generalization. The advantage of multiple layers is that they can learn features at various levels of abstraction (eg CNNs). Aside from the specter of overfitting, the wider your network, the longer it will take to train. If you build a very wide, very deep network, you run the chance of each layer just memorizing what you want the output to be, and you end up with a neural network that fails to generalize to new data.Additionally the number of parameters significantly increase.
<hr/>

6. **The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?**
**Ans**: The universal approximation theorem need to have 2^n nodes in the hidden layer to learn any continuous function, if n is the number of inputs we use. However, if you use lesser number of nodes, the neural network might not necessarily memorize the output for each and every input. [Ref](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6)
<hr/>

7. **What are saddle points and local minima? Which are thought to cause more problems for training large NNs?**
**Ans**: When we optimize neural networks or any high dimensional function, for most of the trajectory we optimize, the critical points(the points where the derivative is zero or close to zero) are saddle points. While local minima are also points where the derivative is zero, but to be a local minimum, it has to be a local minimum in every direction. In contrast, for a saddle point, only 1 direction has to be different than others. Note in higher dimension, the probability of encountering a saddle point is much higher than a local minima.[Ref](https://datascience.stackexchange.com/questions/22853/local-minima-vs-saddle-points-in-deep-learning)</br>
"Saddle Points" as being considered "worse" than "Local Minimums" - this is because "Saddle Points" aren't actually a minimum of any sort, whereas "Local Minimums" are at least minimums at the local level [Ref](https://ai.stackexchange.com/questions/34443/comparing-solutions-from-saddle-points-vs-local-minimums).
<hr/>

8. **Hyperparameters**.</br>
    8.1 **What are the differences between parameters and hyperparameters?** </br>
    **Ans**: Hyperparameters are parameters whose values control the learning process and determine the values of model parameters that a learning algorithm ends up learning. Parameters on the other hand are internal to the model. That is, they are learned or estimated purely from the data during training as the algorithm used tries to learn the mapping between the input features and the labels or targets. </br></br>
    
    8.2 **Why is hyperparameter tuning important?**</br>
    **Ans**:  If we don't correctly tune our hyperparameters, our estimated model parameters produce suboptimal results, as they don't minimize the loss function. This means our model makes more errors</br> </br>
    
    8.3 **Explain algorithm for tuning hyperparameters**.</br>
    **Ans**: Grid search, Random search, Bayesian optimization. [Hyperparameter tuning for machine learning models](https://www.jeremyjordan.me/hyperparameter-tuning/)
    <hr/>
    
9. **Classification vs. regression**</br>
    9.1 **What makes a classification problem different from a regression problem?**</br>
    **Ans**: The most significant difference between regression vs classification is that while regression helps predict a continuous quantity, classification predicts discrete class labels. </br></br>
    
    9.2 **Can a classification problem be turned into a regression problem and vice versa?** </br> 
    **Ans**: Yes, we can do this by a principle called reframing. For example in cases where the output is a continuous value but can have multiple solutions for the same input features, we can treat this problem as a multi-class classification problem instead of a regression problem.</br> [Reframing ML problems](https://towardsdatascience.com/reframing-representing-problems-in-machine-learning-e1805130db29)</br>
<hr/>

10. **Parametric vs. non-parametric methods.**</br>
    10.1 **What’s the difference between parametric methods and non-parametric methods? Give an example of each method.** </br>
    **Ans**: Parametric Methods uses a fixed number of parameters to build the model. Non-Parametric Methods use the flexible number of parameters to build the model. Examples of parametric methods are logisitic regression, linear regression, perceptron, naive bayes, simple neural networks. Examples of non-parametric methods are k-Nearest Neighbours, Decision trees and support vector machines.</br></br>
    
    10.2 **When should we use one and when should we use the other?**</br>
    **Ans**:  Parametric methods make large assumptions about the mapping of the input variables to the output variable and in turn are faster to train, require less data but may not be as powerful.Non-parametric methods make few or no assumptions about the target function and in turn require a lot more data, are slower to train and have a higher model complexity but can result in more powerful models.
    <hr/>
    
 11. **Why does ensembling independently trained models generally improve performance?** </br>
 **Ans**: Ensemble models tend to reduce the variance between model outputs and ground truth. In such methods, each weak learner learns from the probability distribution where the previous weak learner had wrongly predicted results, thus getting better in each iteration.
 Some references for different types of [boosting](https://towardsdatascience.com/advanced-ensemble-learning-techniques-bf755e38cbfb)
<hr/>

12. **Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?**</br>
**Ans**: We can see that with the L2 norm as w gets smaller so does the slope of the norm, meaning that the updates will also become smaller and smaller. When the weights are close to 0 the updates will have become so small as to be almost negligible, so it’s unlikely that the weights will ever become 0. On the other hand, with the L1 norm the slope is constant. This means that as w gets smaller the updates don’t change, so we keep getting the same “reward” for making the weights smaller. Therefore, the L1 norm is much more likely to reduce some weights to 0. The L1 norm will drive some weights to 0, inducing sparsity in the weights. This can be beneficial for memory efficiency or when feature selection is needed (ie we want to select only certain weights). The L2 norm instead will reduce all weights but not all the way to 0. This is less memory efficient but can be useful if we want/need to retain all parameters. [Ref](https://towardsdatascience.com/visualizing-regularization-and-the-l1-and-l2-norms-d962aa769932, https://satishkumarmoparthi.medium.com/why-l1-norm-creates-sparsity-compared-with-l2-norm-3c6fa9c607f4)
<hr/>

13. **Why does an ML model’s performance degrade in production?**</br>
**Ans**: Over time many models' predictive performance decreases as a given model is tested on new datasets within rapidly evolving environments
<hr/>

14. **What problems might we run into when deploying large machine learning models?** </br>
**Ans**: 
    * High resource availability and throughput (For eg online recommendation systems). Keep some precomputed predictions/embeddings in a cache for faster computation.
     * Silent failures and system failures. Using continuous model monitoring
     * Correlation of offline and online metrics for the same model. Solution: Using conterfactual evaluations.
     * Model size and scale before and after deployment. Solution: Accessig models through endpoints, using kubernetes clusters to scale model operations.
     Read more about this in [Model Deployment Challenges: 6 Lessons From 6 ML Engineers](https://neptune.ai/blog/model-deployment-challenges-lessons-from-ml-engineers)
     Problems in ML Lifecycle: [Challenges Deploying Machine Learning Models to Production](https://towardsdatascience.com/challenges-deploying-machine-learning-models-to-production-ded3f9009cb3)
     <hr/>
     
15. **Your model performs really well on the test set but poorly in production. What are your hypotheses about the causes? How do you validate whether your hypotheses are correct? Imagine your hypotheses about the causes are correct. What would you do to address them?**</br>
**Ans**:
    * [Why Production Machine Learning Fails — And How To Fix It ](https://www.montecarlodata.com/blog-why-production-machine-learning-fails-and-how-to-fix-it/)</br>
    * [9 Reasons why Machine Learning models not perform well in production](https://towardsdatascience.com/9-reasons-why-machine-learning-models-not-perform-well-in-production-4497d3e3e7a5)
    * [5 Must-Do Error Analysis Before You Put Your Model in Production ](https://neptune.ai/blog/must-do-error-analysis)
    
 
 
    
    
    

    

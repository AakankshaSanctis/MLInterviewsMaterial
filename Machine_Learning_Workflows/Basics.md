# Basics


1. *Explain supervised, unsupervised, weakly supervised, semi-supervised, reinforcement learning, self-supervised learning and active learning* </br>
* Supervised learning: Supervised learning (SL) is a machine learning paradigm for problems where the available data consists of labelled examples, meaning that each data point contains features (covariates) and an associated label. The goal of supervised learning algorithms is learning a function that maps feature vectors (inputs) to labels (output), based on example input-output pairs.  
* Unsupervised learning: Unsupervised learning is a type of algorithm that learns patterns from untagged data. Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data. Two types of algorithms are used: Clustering and Association.
* Weakly supervised learning: Weak supervision is a branch of machine learning where noisy, limited, or imprecise sources are used to provide supervision signal for labeling large amounts of training data in a supervised learning setting. This approach alleviates the burden of obtaining hand-labeled data sets, which can be costly or impractical. Eg. Global statistics, Weak classifiers etc.
* Semi-supervised learning: Semi-supervised learning is a learning problem that involves a small number of labeled examples and a large number of unlabeled examples. Learning problems of this type are challenging as neither supervised nor unsupervised learning algorithms are able to make effective use of the mixtures of labeled and untellable data. As such, specialized semis-supervised learning algorithms are required. The primary difference, though, is that semi-supervised learning propagates knowledge (“based on what is already labeled, label some more”) whereas weak supervision injects knowledge (“based on your knowledge, label some more”) [Some heuristics]
* Active learning: Active learning is the subset of machine learning in which a learning algorithm can query a user interactively to label data with the desired outputs. In active learning, the algorithm proactively selects the subset of examples to be labeled next from the pool of unlabeled data. The fundamental belief behind the active learner algorithm concept is that an ML algorithm could potentially reach a higher level of accuracy while using a smaller number of training labels if it were allowed to choose the data it wants to learn from. Three types of active learning are: Stream-based selective sampling, pool-based sampling and membership query synthesis. 
* Reinforcement learning: Reinforcement Learning(RL) is a type of machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences. Though both supervised and reinforcement learning use mapping between input and output, unlike supervised learning where the feedback provided to the agent is correct set of actions for performing a task, reinforcement learning uses rewards and punishments as signals for positive and negative behavior
* Self-supervised learning: Self-Supervised Learning (SSL) is a Machine Learning paradigm where a model, when fed with unstructured data as input, generates data labels automatically, which are further used in subsequent iterations as ground truths. The fundamental idea for self-supervised learning is to generate supervisory signals by making sense of the unlabeled data provided to it in an unsupervised fashion on the first iteration. Then, the model uses the high confidence data labels among those generated to train the model in the next iterations like any other supervised learning model via backpropagation. The only difference is, the data labels used as ground truths in every iteration are changed. 
<hr/>

2. What is [emphirical risk minimization](https://towardsdatascience.com/learning-theory-empirical-risk-minimization-d3573f90ff77)? </br>
Ans: The difference between the emphirical error (on a sample S sampled from distribution D) and the true error. Ideally, we want the difference between these two errors to be minimum.</br></br>
    2.1 What’s the risk in empirical risk minimization? </br>
    Ans: The error is called the risk in emphirical risk minimization.</br></br>
    2.2 Why is it emphirical?</br>
    Ans: Because it is computed on a sample of the original distribution and not on the              entire distribution.</br></br>
    2.3 How do we minimize that risk? </br>
    Ans: Avoid overfitting, this will decrease the emphirical error but significantly                increase the true error. Using regularization (L2) will help in this case.                  Increasing the sample size will also decrease the emphirical and true error. 
    <hr/>
    
 3. Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML? </br>
 Ans:  If two models have the same performance on the validation/testing dataset select the simpler model because it is more likely to generalize well. Occam’s Razor is really just an example of the bias-variance tradeoff in machine learning. When selecting a model to use for any problem, we want a model that is complex enough to avoid underfitting, and simple enough to avoid overfitting. [Ref](https://towardsdatascience.com/what-occams-razor-means-in-machine-learning-53f07effc97c)
 <hr/>
 
 4. What are the conditions that allowed deep learning to gain popularity in the last decade? </br>
 Ans: The availability of massive amounts of data for training computer systems. Availability of GPU cluster on cloud and distributed processes.
 <hr/>
 
 5. If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why? </br>
 Ans: The main issue is that very wide, shallow networks are very good at memorization, but not so good at generalization. The advantage of multiple layers is that they can learn features at various levels of abstraction (eg CNNs). Aside from the specter of overfitting, the wider your network, the longer it will take to train. If you build a very wide, very deep network, you run the chance of each layer just memorizing what you want the output to be, and you end up with a neural network that fails to generalize to new data.Additionally the number of parameters significantly increase.
<hr/>

6. The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?
Ans: The universal approximation theorem need to have 2^n nodes in the hidden layer to learn any continuous function, if n is the number of inputs we use. However, if you use lesser number of nodes, the neural network might not necessarily memorize the output for each and every input. [Ref](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6)
 
    

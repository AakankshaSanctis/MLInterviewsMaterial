
1.**What are the basic assumptions to be made for linear regression?** </br>
**Ans:** 
      * Linear relationship between dependent variable (Y) and independent variable (X)
      * Residuals are normally distributed
      * Variance across residuals is constant for all independent variables (Homoscedasticity)
      * No autocorrelation between errors
      * No correlation between independent variables (Multicollinearity)
      More reading: https://towardsdatascience.com/linear-regression-assumptions-why-is-it-important-af28438a44a1
      <hr/>
      
2.**What happens if we don’t apply feature scaling to logistic regression?** </br>
**Ans:** Logistic Regression uses Gradient Descent as one of the approaches for obtaining the best result, and feature scaling helps to speed up the Gradient Descent convergence process. When we have features that vary greatly in magnitude, the algorithm assumes that features with a large magnitude are more relevant than those with a small magnitude. As a result, when we train the model, those characteristics become more important.
Because of this feature scaling is required to put all features into the same range, regardless of their relevance. 
<hr/>
 
3.**What are the algorithms you’d use when developing the prototype of a fraud detection model?** </br>
**Ans:** Frauds are usually outliers, so clustering techiques work well to detect frauds. Classification models such as logistic regression, random forests or MLPs can also be used for fraud detection. Proximity methods such as K-NNs also help in anamoly detection. Note that the dataset is highly imbalanced in case of fraud detection, so you can use weighted loss metric to give the minority class more weightage (Upsampling might give bad results as it simulates fake examples)
A more detailed working of creating this model by Nvidia's team: https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
<hr/>

4. **Why do we use feature selection?** </br>
**Ans:** Feature selection techniques are employed to reduce the number of input variables by eliminating redundant or irrelevant features. It then narrows the set of features to those most relevant to the machine learning model.
Three key benefits of feature selection are:
* Decreases over-fitting  
  * Fewer redundant data means fewer chances of making decisions based on noise.
* Improves Accuracy  
   * Less misleading data means better modeling accuracy.
* Reduces Training Time  
   * Less data means quicker algorithms.
 <hr/>
 
 5.**What are some of the algorithms for feature selection? Pros and cons of each.** </br>
 **Ans:** Wrapper methods are likely to overfit to the model type, and the feature subsets they produce might not generalize should one want to try them with a different model. Another significant disadvantage of wrapper methods is their large computational needs. They require training a large number of models, which might require some time and computing power. 
 * Backward selection, in which we start with a full model comprising all available features. In subsequent iterations, we remove one feature at a time, always the one that yields the largest gain in a model performance metric, until we reach the desired number of features.
 * Forward selection, which works in the opposite direction: we start from a null model with zero features and add them greedily one at a time to maximize the model’s performance.
 * Recursive Feature Elimination, or RFE, which is similar in spirit to backward selection. It also starts with a full model and iteratively eliminates the features one by one. The difference is in the way the features to discard are chosen. Instead of relying on a model performance metric from a hold-out set, RFE makes its decision based on feature importance extracted from the model. This could be feature weights in linear models, impurity decrease in tree-based models, or permutation importance (which is applicable to any model type).
Check out this awesome article on Feature selection: https://neptune.ai/blog/feature-selection-methods
<hr/>

6. **K-means clustering: How would you choose the value of k?** </br>
**Ans:** We can use the elbow method and the silhouette method to choose the most optimal k given a range of k's. More detailed explaination: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
<hr/>

7. **If the labels are known, how would you evaluate the performance of your k-means clustering algorithm?**</br>
**Ans:** Measures like purity (align max overlap between cluster and classes and evaluate accuracy), randIndex(check if pair of instances fall in the same cluster if they belong to the same class and calculate accuracy, F1, precision, recall) and normalized mutual information. Detailed formulas: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
<hr/>

8. **How would you do it if the labels aren’t known?** </br>
**Ans:**:
     * Silhoeutte coefficient (Score = 1 means good clustering while -1 means really bad). Checks how similar instance is to assigned cluster compared to other clusters.
     * Calinski-Harabasz Index (Variance Ratio Criterion): Ratio of the squared inter-cluster distance sum and the squared intra-cluster distance sum for all clusters. Higher CH is, the better the clusters are separated from each other, and there’s no upper bound for CH as those for Silhouette score.
     More metrics: https://towardsdatascience.com/three-performance-evaluation-metrics-of-clustering-when-ground-truth-labels-are-not-available-ee08cb3ff4fb
<hr/>

9. **Given the following dataset, can you predict how K-means clustering works on it? Explain.**</br>
![image](https://user-images.githubusercontent.com/29446732/214691886-2591f5a1-dbcf-45a5-a4f9-510f5c5a3b81.png)
**Ans:** k means clustering requires linear boundaries. Since k means clustering tries to find the mean point as the cluster centroid, in the case of concentric circle, the mean for both the clusters will lie at the same point. Therefore for k>1 , it can never predict the circles as two different clusters.
![image](https://user-images.githubusercontent.com/29446732/214695966-70a9547c-da86-4a39-891c-16d02288b919.png)</br>
Spectral clustering is a technique that attempts to overcome the linear boundary problem of k-means clustering. It works by treating clustering as a graph partitioning problem, its looking for nodes in a graph with a small distance between them. Spectral clustering uses something called a kernel trick to introduce additional dimensions to the data. Spectral clustering will introduce an additional dimension that effectively moves one of the circles away from the other in the additional dimension. This has the downside of being more computationally expensive than k-means clustering. 
![image](https://user-images.githubusercontent.com/29446732/214696248-965d3814-5ec7-4953-940e-2fe0b5ea20e3.png)
How to implement: https://scw-aberystwyth.github.io/machine-learning-novice/04-clustering/index.html
<hr/>

10. **k-nearest neighbor classification: How would you choose the value of k?** </br>
**Ans:**
     * There are no pre-defined statistical methods to find the most favorable value of K.
     * Initialize a random K value and start computing.
     * Choosing a small value of K leads to unstable decision boundaries.
     * The substantial K value is better for classification as it leads to smoothening the decision boundaries.
     * Plot the error rate on a validation set vs the number of k and choose the k with the smallest error
     * Choose odd k's to not have any ties.
<hr/>

11. **What happens when you increase or decrease the value of k?**
**Ans**: Large values of k will lead to everything being classified as the most probable class. Small values of k will make the model highly variable and unstable, leading to large changes in classification for small changes in the data. 
<hr/>

12. **How does the value of k impact the bias and variance?** </br>
**Ans:** Large k's lead to high bias, low variance while small k lead to low bias and high variance.
<hr/>

13. **K-means vs GMM, compare the two and when would you choose one over another?** </br>
**Ans:** A good playlist for Gaussian mixture models and the EM algorithm: https://www.youtube.com/watch?v=3JYcCbO5s6M&list=PLBv09BD7ez_7beI0_fuE96lSbsr_8K8YD&ab_channel=VictorLavrenko </br>
* The first visible difference between K-Means and Gaussian Mixtures is the shape the decision boundaries. GMs are somewhat more flexible and with a covariance matrix ∑ we can make the boundaries elliptical, as opposed to circular boundaries with K-means.
* Another thing is that GMs is a probabilistic algorithm. By assigning the probabilities to datapoints, we can express how strong is our belief that a given datapoint belongs to a specific cluster.
* If we compare both algorithms, the Gaussian mixtures seem to be more robust. However, GMs usually tend to be slower than K-Means because it takes more iterations of the EM algorithm to reach the convergence. They can also quickly converge to a local minimum that is not a very optimal solution.</br>

If you look for robustness, GM with K-Means initializer seems to be the best option. K-Means should be theoretically faster if you experiment with different parameters. GM on its own is not much of use because it converges too fast to a non-optimal solution for this dataset.
<hr/>

14. **Bagging and boosting are two popular ensembling methods. Random forest is a bagging example while XGBoost is a boosting example. What are some of the fundamental differences between bagging and boosting algorithms?** </br>
**Ans** Detailed overview: https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

15. **How are the above ensemble methods used in deep learning? ** </br>
**Ans:** https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/

16. **Imagine we build a user-item collaborative filtering system to recommend to each user items similar to the items they’ve bought before. You can build either a user-item matrix or an item-item matrix. What are the pros and cons of each approach?** </br>
**Ans:** 
* Item-Item: 
     * To make a new recommendation to a user, the idea of item-item method is to find items similar to the ones the user already “positively” interacted with. Two items are considered to be similar if most of the users that have interacted with both of them did it in a similar way.
* User-User: 
     * In order to make a new recommendation to a user, user-user method roughly tries to identify users with the most similar “interactions profile” (nearest neighbours) in order to suggest items that are the most popular among these neighbours (and that are “new” to our user). </br>

The user-user method is based on the search of similar users in terms of interactions with items. As, in general, every user have only interacted with a few items, it makes the method pretty sensitive to any recorded interactions (high variance). On the other hand, as the final recommendation is only based on interactions recorded for users similar to our user of interest, we obtain more personalized results (low bias). </br>

Conversely, the item-item method is based on the search of similar items in terms of user-item interactions. As, in general, a lot of users have interacted with an item, the neighbourhood search is far less sensitive to single interactions (lower variance). As a counterpart, interactions coming from every kind of users (even users very different from our reference user) are then considered in the recommendation, making the method less personalised (more biased). Thus, this approach is less personalized than the user-user approach but more robust.

Detailed reading on recsys: https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada
<hr/>

17. **How to handle cold-start problem in recommendation systems?** </br>
**Ans:** 
* New item cold start: Use content based filtering. New item will have its own set of features
* New visitor cold start: 
     * Apply popularity based strategy
     * Averaging over users wrt properties such as geo-location, device etc.</br>
 Relevant read: https://medium.com/@markmilankovich/the-cold-start-problem-for-recommender-systems-89a76505a7
 <hr/>
  
18. **How is Naive Bayes classifier naive?** </br>
**Ans** Its naive because it assumes independence between features even though they might not be.
<hr/>

19. **What is gradient boosting?** </br>
**Ans:** Build trees in iterations based on the error(residuals) produced by the last tree. In the first iteration, take the average of all target values and calculate residuals for each instance. Now choose the features, build the tree, and predict the residuals. (You can fix the number of leaves). To avoid overfitting, use lr (new prediction = old prediction + lr* leaf value prediction). Ultimately the residuals should minimize and the model converges. </br>
Must watch series: https://www.youtube.com/watch?v=3CC4N4z3GJc&ab_channel=StatQuestwithJoshStarmer</br>
Reading: https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502

 
* Reminder: Check the math of GBM in detail
<hr/>

20. **What problems is gradient boosting good for?**
**Ans** Gradient Boosting Algorithm is generally used when we want to decrease the Bias error. ii) Gradient Boosting Algorithm can be used in regression as well as classification problems. In regression problems, the cost function is MSE whereas, in classification problems, the cost function is Log-Loss.
<hr/>

21. **XGBoost and LightGBM**
**Ans:** https://neptune.ai/blog/xgboost-vs-lightgbm




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


 
 



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
 
 


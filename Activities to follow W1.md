# Possible activities or tasks to be done
+ Directories structure (can be changed)
  + data: directory with datasets
  + utils: directory with .py scripts containing useful functions or algorithms implementations
    + data_preparation.py: only data preparation functions
    + algorithms.py: Implemented algorithms functions as K-means and a variation, and probably metrics too
    + visualization.py: for visualization purposes
  + main.py: main script with entire pipeline
  
+ Tasks to be done (complete or change if needed)
  + Unify data preparation in single script and organize functions accordingly to directory structure
    + Save target variable for last point
  + K means algorithm (as a class or a function?)
    + input: pandas dataframe with already prepared and normalized selected variables (or numpy array?)
    1. Initialization function
    2. Function to associate each point to a cluster
    3. Re-estimation function for the clusters
    4. Main function running loops and checking threshold or number of iterations
    5. Metric function
  + K means variation (similar to previous one but with additional functions). Better if made after finishing k means to share structure
  + DBSCAN
    + Parameter estimation analysis
  + Bisecting K Means
  + Fuzzy clustering algorithm
  + Elbow diagram
  + Iterations to be done and criteria to select the best one
    + Vary K, initialization seed, initialization method (if needed)
  + Metrics to compare: "Adjusted Rand Index, Purity, Davies–Bouldin index, F-measure. You can use these ones or other ones from the literature that best suit
your evaluation. For the evaluation metrics, you can use the ones defined in sklearn
library."
  + Compare to true values: "The results obtained can be compared to the true values. To show the results, you can use a
confusion matrix, for example."
    
 + Considerations:
   + Constantly thinking or writing as a draft possible useful conclusions and creating visualizations for the document
   + Algorithms differences analysis in output, weaknesses, advantages, execution time, variable types, etc
  
    "- Which information can be obtained for each data set using each algorithm? Is it the same or
    not?
    - Which clustering algorithm do you consider is the best one for datasets with categorical data,
    with numerical data and with mixed data?
    - Did you find differences among algorithms? According to the data sets chosen, which
    algorithm gives you more advice for knowing the underlying information in the data set?
    - Can you explain the setup that you have used for each algorithm?
    - In the case of the K-Means and the other algorithms where you have to choose the K, which
    has been the best K value? Have you implemented any improvement on the basic
    algorithms? For example, you can introduce/use a performance measure, like Silhouette, to
    decide which the best K value is. Another example, in K-Means++ it is defined an algorithm
    for choosing the initial values (or "seeds") for the k-means clustering algorithm. Alternatively,
    you can use G-Means that is an algorithm for determining automatically the best K value.
    - In the case of Fuzzy Clustering algorithm, you can optimize the C value. Have you done the
    optimization? Which are the results? In case that you have not included the optimization,
    how many C- values have you tested for each data set? And which value do you consider it
    is the best one?"
    

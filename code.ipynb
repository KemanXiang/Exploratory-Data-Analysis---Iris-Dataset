# %% [markdown]
# **Loading the Iris dataset from Scikit-learn**

# %% [code] {"jupyter":{"outputs_hidden":true}}
# import load_iris function from datasets module
from sklearn.datasets import load_iris

# %% [markdown]
# **Data as table**
# 
# A basic table is a two-dimensional grid of data, in which the rows represent individual elements of the dataset, and the columns represent quantities related to each of these elements. In general, we will refer to the rows of the matrix as samples, and the number of rows as n_samples and the the columns of the matrix as features, and the number of columns as n_features.
# 
# **Features matrix** - This table layout makes clear that the information can be thought of as a two-dimensional numerical array or matrix,  called  the features matrix with shape [n_samples, n_features]
# 
# **Target array.**- In addition to the feature matrix X, we also generally work with a label or target array, which by convention we will usually call y. The target array is usually one dimensional, with length n_samples, and is generally contained in a NumPy array or Pandas Series.

# %% [code]
# save "bunch" object containing iris dataset and iits attributes
iris = load_iris()
type(iris) 

# %% [code]
#print the iris dataset
# Each row represents the flowers and each column represents the length and width.
print (iris.data)
iris.data.shape

# %% [markdown]
# **Machine Learning Terminology**
# 
# 1.  Each row is  an **observation** (also known as : sample, example, instance, record)
# 
# 2. Each column is a **feature** (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate)

# %% [code]
# print the names of the four features
print (iris.feature_names)

# %% [code]
# print the integers representing the species of each observation
print (iris.target)

# %% [code]
# print the encoding scheme for species; 0 = Setosa , 1=Versicolor, 2= virginica
print (iris.target_names)

# %% [markdown]
# Each value we are predicting is the **response** (also known as: target, outcome, label, dependent variable)
# 
# **Classification** is supervised learning in which the response is categorical
# 
# **Regression** is supervised learning in which the response is ordered and continuous

# %% [markdown]
# **Requirements for working with data in scikit-learn**
# 
# 1) Features  and response are **separate objects**
# 
# 2) Features and response should be **numeric**
# 
# 3)Features and response should be **NumPy arrays**
# 
# 4)Features and response should have **specific shapes**

# %% [code]
# Check the types of the features and response
type('iris.data')
type('iris.target')

# %% [code]
# Check the shape of the features 
#(first dimension = (ROWS) ie number of observations, second dimensions = (COLUMNS) ie number of features)
iris.data.shape

# %% [code]
# Check the sape of the response (single dimension matching the number of observation)
iris.target.shape

# %% [markdown]
# **1. Scatter Plot with Iris Dataset ** 

# %% [code]
# Extract the values for features and create a list called featuresAll
featuresAll=[]
features = iris.data[: , [0,1,2,3]]
features.shape

# %% [code]
# Extract the values for targets
targets = iris.target
targets.reshape(targets.shape[0],-1)
targets.shape

# %% [code]
# Every observation gets appended into the list once it is read. For loop is used for iteration process
for observation in features:
    featuresAll.append([observation[0] + observation[1] + observation[2] + observation[3]])
print (featuresAll)


# %% [code]
# Plotting the Scatter plot
import matplotlib.pyplot as plt
plt.scatter(featuresAll, targets, color='red', alpha =1.0)
plt.rcParams['figure.figsize'] = [10,8]
plt.title('Iris Dataset scatter Plot')
plt.xlabel('Features')
plt.ylabel('Targets')


# %% [markdown]
# **1a) Scatter Plot with Iris Dataset (Relationship between Sepal Length and Sepal Width) # Method 1**

# %% [code]
#Finding the relationship between Sepal Length and Sepal width
featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[0]) #Sepal length
    targets.append(feature[1]) #sepal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    #item = (featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), (featuresAll[100:150], targets[100:150])
    x, y = item
    plt.scatter(x, y,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('sepal length')
plt.ylabel('Sepal width')
plt.show()


# %% [markdown]
# **1b) Scatter Plot with Iris Dataset (Relationship between Petal Length and Petal Width) # Method 1  **

# %% [code]
#Finding the relationship between Petal Length and Petal width
featuresAll = []
targets = []
for feature in features:
    featuresAll.append(feature[2]) #Petal length
    targets.append(feature[3]) #Petal width

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), 
        (featuresAll[100:150], targets[100:150]))

for item, color, group in zip(data,colors,groups): 
    #item = (featuresAll[:50], targets[:50]), (featuresAll[50:100], targets[50:100]), (featuresAll[100:150], targets[100:150])
    x0, y0 = item
    plt.scatter(x0, y0,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()

# %% [markdown]
#   **2. K - Nearest Neighbours (KNN) Algorithm**

# %% [markdown]
# **sklearn.neighbors** provides functionality for unsupervised and supervised neighbors-based learning methods. **Supervised neighbors-based learning** comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels. **Unsupervised nearest neighbors** is the foundation of many other learning methods, notably manifold learningand spectral clustering.
# 
# Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems, including handwritten digits or satellite image scenes. Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.

# %% [code]
import pandas as pd
iris = load_iris()
ir = pd.DataFrame(iris.data)
ir.columns = iris.feature_names
ir['CLASS'] = iris.target
ir.head()

# %% [markdown]
# The classes in **sklearn.neighbors** can handle either Numpy arrays or scipy.sparse matrices as input. For dense matrices, a large number of possible distance metrics are supported.

# %% [code]
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(5) #The arguements specify to return the Fast 5 most among the dataset 
nn.fit(iris.data)

# %% [code]
ir.describe()

# %% [code]
#creating a test data
import numpy as np
test = np.array([5.4,2,2,2.3])
test1 = test.reshape(1,-1)
test1.shape

# %% [code]
nn.kneighbors(test1,5)

# %% [code] {"scrolled":true}
ir.ix[[98, 93, 57, 60, 79],]

# %% [markdown]
# **3. KNeighborsClassifier Algorithm**

# %% [code]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()

# %% [markdown]
# **KNN Classifiers Algorithm - How it works? - With Easy explanation**

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.neighbors import KNeighborsClassifier

# %% [code] {"jupyter":{"outputs_hidden":true}}
knn = KNeighborsClassifier(n_neighbors=1)

# %% [code]
print (knn)

# %% [code] {"jupyter":{"outputs_hidden":true}}
import numpy as np
X1 = np.asarray(featuresAll)
X1 = X1.reshape(-1,1)

# %% [code]
X1.shape

# %% [code]
y = iris.target

y.shape

# %% [code] {"scrolled":true}
knn.fit(X1, y)

# %% [code]
import numpy as np
print (knn.predict([[6.4]]))

# %% [code] {"jupyter":{"outputs_hidden":true}}
knn = KNeighborsClassifier(n_neighbors=5)

# %% [code]
knn.fit(X1, y)

# %% [code]
print (knn.predict([[3.4]]))

# %% [code]
print (knn.predict(np.column_stack([[1.,6.1,3.2,4.2]])))

# %% [markdown]
# **Linear regression**

# %% [markdown]
# We will start with the most familiar linear regression, a straight-line fit to data. A straight-line fit is a model of the form
# y=ax+b
# where a is commonly known as the slope, and b is commonly known as the intercept.
# 
# We can use Scikit-Learn's LinearRegression estimator to fit this data and construct the best-fit line:

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.linear_model import LinearRegression

# %% [code]
model = LinearRegression(fit_intercept=True)
model

# %% [code]
import numpy as np
XX = np.asarray(featuresAll)
X2 = XX[:, np.newaxis]
X2
X2.shape

# %% [code]
y2 = iris.target
y2.shape


# %% [code]
model.fit(X2, y2)

# %% [markdown]
# The slope and intercept of the data are contained in the model's fit parameters, which in Scikit-Learn are always marked by a trailing underscore. Here the relevant parameters are coef_ and intercept_:

# %% [code]
model.coef_

# %% [code]
model.intercept_

# %% [code]
Xfit = np.random.randint(8,size=(150))
Xfit.astype(float)
Xfit = Xfit[:, np.newaxis]
Xfit.shape

# %% [code]
yfit = (model.predict(Xfit))
yfit.shape

# %% [code]
plt.scatter(X2, y2)
plt.plot(Xfit, yfit)

# %% [markdown]
# **Regression**
# 
# In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships among variables. It includes many techniques for modeling and analyzing several variables, when the focus is on the relationship between a dependent variable and one or more independent variables (or 'predictors'). More specifically, regression analysis helps one understand how the typical value of the dependent variable (or 'criterion variable') changes when any one of the independent variables is varied, while the other independent variables are held fixed.
# 
# One trick you can use to adapt linear regression to nonlinear relationships between variables is to transform the data according to basis functions. We have seen one version of this before, in the PolynomialRegression pipeline used in Hyperparameters and Model Validation and Feature Engineering. The idea is to take our multidimensional linear model:
# y=a0+a1x1+a2x2+a3x3+⋯
# and build the x_1, x_2, x_3, and so on, from our single-dimensional input x. 

# %% [markdown]
# This polynomial projection is useful enough that it is built into Scikit-Learn, using the PolynomialFeatures transformer:

# %% [code]
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(150, include_bias=False)
poly.fit_transform(X2)

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(3),
                           LinearRegression())
poly_model.fit(X2, y2)
yfit = poly_model.predict(Xfit)

# %% [code]
#Our linear model, through the use of 3rd-order polynomial basis functions, can provide a fit to this non-linear data
plt.scatter(X2, y2)
plt.plot(Xfit, yfit);

# %% [markdown]
# **How the length and width vary according to the species**

# %% [code]
import pandas as pd
iris1 = pd.read_csv("../input/Iris.csv") #load the dataset
iris1.head(5)

# %% [markdown]
# **1c) Scatter Plot with Iris Dataset (Relationship between Sepal Length and SepalWidth) # Method 1  **

# %% [code]
iris1.plot(kind ='scatter', x ='SepalLengthCm', y ='SepalWidthCm')
plt.show()

# %% [markdown]
# **1d) Scatter Plot with Iris Dataset (Relationship between Petal Length and Petal Width) Method 1  **

# %% [code]
iris1.plot(kind ='scatter', x ='PetalLengthCm', y ='PetalWidthCm')
plt.show()

# %% [markdown]
# **Histograpm Plot of Iris Data **

# %% [code] {"scrolled":true}
exclude = ['Id']
iris1.ix[:, iris1.columns.difference(exclude)].hist() 
plt.figure(figsize=(15,10))
plt.show()

# %% [markdown]
# **Violin Plot**

# %% [code]
import seaborn as sns
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris1)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris1)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris1)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=iris1)

# %% [markdown]
# Now, when we train any algorithm, the number of features and their correlation plays an important role. If there are features and many of the features are highly correlated, then training an algorithm with all the featues will reduce the accuracy. Thus features selection should be done carefully. This dataset has less featues but still we will see the correlation.

# %% [markdown]
# **IRIS Correlation Matrix**

# %% [code]
corr = iris1.corr()
corr

# %% [code]
# import correlation matrix to see parametrs which best correlate each other
# According to the correlation matrix results Petal LengthCm and
#PetalWidthCm have positive correlation which is proved by the scatter plot discussed above

import seaborn as sns
import pandas as pd
corr = iris1.corr()
plt.figure(figsize=(10,8)) 
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap='viridis', annot=True)
plt.show()

# %% [markdown]
# **Supervised learning example: Iris classification**

# %% [code] {"jupyter":{"outputs_hidden":true}}
# I prefer to use train_test_split for cross-validation
# This piece will prove us if we have overfitting 
X3 = iris1.iloc[:, 0:5]  
Y3 = iris1['Species']

# %% [markdown]
# We would like to evaluate the model on data it has not seen before, and so we will split the data into a training set and a testing set. This could be done by hand, but it is more convenient to use the **train_test_split** utility function

# %% [code]
from sklearn.cross_validation import train_test_split
X3_train, X3_test, y_train, y_test = train_test_split(X3, Y3, test_size=0.4, random_state=0)
print(" X3_train",X3_train)
print("X3_test",X3_test)
print("y_train",y_train)
print("y_test",y_test)

# %% [markdown]
# **With the data arranged, we can follow our recipe to predict the labels:**

# %% [code]
#Train and test model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model = model.fit(X3_train ,y_train)
y_model = model.predict(X3_test)
y_model

# %% [markdown]
# Finally, we can use the **accuracy_score** utility to see the fraction of predicted labels that match their true value:                 

# %% [code]
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_model) 

# %% [markdown]
# With an accuracy topping 96%, we see that even this very naive classification algorithm is effective for this particular dataset!

# %% [markdown]
# ** K Means Clustering in SciKit Learn with Iris Data**

# %% [markdown]
# k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. 
# 
# The k-means algorithm searches for a pre-determined number of clusters within an unlabeled multidimensional dataset. It accomplishes this using a simple conception of what the optimal clustering looks like:
# 
# The "cluster center" is the arithmetic mean of all the points belonging to the cluster.
# Each point is closer to its own cluster center than to other cluster centers. Those two assumptions are the basis of the k-means model. 

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.cluster import KMeans

# %% [code] {"jupyter":{"outputs_hidden":true}}
km = KMeans(n_clusters=3, max_iter =1000)

# %% [code] {"scrolled":true}
X1.shape

# %% [code]
km.fit(iris.data)

# %% [code]
km.cluster_centers_

# %% [code]
km.labels_

# %% [code] {"scrolled":true}
iris1[' K Mean predicted label'] = km.labels_
iris1

# %% [code] {"scrolled":true}
#First, let's generate a two-dimensional dataset containing four distinct blobs. 
#To emphasize that this is an unsupervised algorithm, we will leave the labels out of the visualization.
from sklearn.datasets.samples_generator import make_blobs
X1, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X1[:, 0], X1[:, 1], s=50);

# %% [code] {"jupyter":{"outputs_hidden":true}}
#By eye, it is relatively easy to pick out the four clusters. 
#The k-means algorithm does this automatically, and in Scikit-Learn uses the typical estimator API:
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X1)
y_kmeans = kmeans.predict(X1)

# %% [code]
#Let's visualize the results by plotting the data colored by these labels.
#We will also plot the cluster centers as determined by the k-means estimator:
plt.scatter(X1[:, 0], X1[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

# %% [markdown]
# **Unsupervised learning example: Iris dimensionality**

# %% [markdown]
# As an example of an unsupervised learning problem, let's take a look at reducing the dimensionality of the Iris data so as to more easily visualize it. Recall that the Iris data is four dimensional: there are four features recorded for each sample.
# 
# The task of dimensionality reduction is to ask whether there is a suitable lower-dimensional representation that retains the essential features of the data. Often dimensionality reduction is used as an aid to visualizing data: after all, it is much easier to plot data in two dimensions than in four dimensions or higher!
# 
# Principal component analysis- PCA which is a fast linear dimensionality reduction technique. 

# %% [code] {"jupyter":{"outputs_hidden":true}}
from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)  # 2. Instantiate the model with hyperparameters 

# %% [code]
model.fit(X3) 

# %% [code]
X_2D = model.transform(X3) # 3. Fit to data. Notice y is not specified!
X_2D

# %% [code]
X_2D.shape # 4. Transform the data to two dimensions

# %% [code]
X_2D[:, 0]

# %% [code]
X_2D[:, 1]

# %% [code]
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)

# %% [markdown]
# **Pivot the Data with Iris Dataset**

# %% [code]
import pandas as pd
iris1 = pd.read_csv("../input/Iris.csv") #load the dataset
iris1.head(10)

# %% [markdown]
# **The simplest pivot table must have a dataframe and an index . In this case, let’s use the Species as our index.**

# %% [code]
pd.pivot_table(iris1,index=["Id"])

# %% [markdown]
# **You can have multiple indexes as well. In fact, most of the pivot_table args can take multiple values via a list.**

# %% [code]
pd.pivot_table(iris1,index=["Id","Species"])

# %% [markdown]
# **This is interesting but not particularly useful. What we probably want to do is look at this by  Species and ID. It’s easy enough to do by changing the index .**

# %% [code]
pd.pivot_table(iris1,index=["Species","Id"])

# %% [markdown]
# **You can see that the pivot table is smart enough to start aggregating the data and summarizing  Sepal Lenth and Petal length  with their Species name.** 

# %% [code]
pd.pivot_table(iris1,index=["Species"],values=["SepalLengthCm","SepalWidthCm"])

# %% [markdown]
# **The SepalLength and SepalWidth column automatically averages the data but we can do a count or a sum.**

# %% [code]
pd.pivot_table(iris1,index=["Species"],values=["SepalLengthCm","SepalWidthCm"],aggfunc=np.sum)

# %% [markdown]
# **aggfunc can take a list of functions. Let’s try a mean using the numpy mean function and len to get a count.**

# %% [code]
pd.pivot_table(iris1,index=["Species"],values=["SepalLengthCm","SepalWidthCm"],aggfunc=[np.mean,len])

# %% [code]
pd.pivot_table(iris1,index=["Species"],values=["SepalLengthCm","SepalWidthCm"],
               columns=["PetalLengthCm"],aggfunc=[np.sum])

# %% [markdown]
# **The NaN’s are a bit distracting. If we want to remove them, we could use fill_value to set them to 0.**

# %% [code]
pd.pivot_table(iris1,index=["Species"],values=["SepalLengthCm","SepalWidthCm"],
               columns=["PetalLengthCm"],aggfunc=[np.sum],fill_value=0)

# %% [markdown]
# **Add Sepal Width to the index list.**

# %% [code]
pd.pivot_table(iris1,index=["Species","SepalLengthCm","SepalWidthCm","PetalWidthCm"],
               values=["PetalLengthCm"],aggfunc=[np.sum],fill_value=0)

# %% [markdown]
# For this data set, this representation makes more sense. Now, what if I want to see some totals? margins=True does that for us.

# %% [code]
df = pd.pivot_table(iris1,index=["Species","SepalLengthCm","SepalWidthCm","PetalWidthCm"],
               values=["PetalLengthCm"],aggfunc=[np.sum,np.mean],fill_value=0,margins=True)
df

# %% [markdown]
# Suppose, If you want to look at just one Species:

# %% [code]
df.query('Species == ["Iris-virginica"]')

# %% [code] {"jupyter":{"outputs_hidden":true}}

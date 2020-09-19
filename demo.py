# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <br><br><br><br><br><h1 style="font-size:4em;color:#2467C0">Welcome to Week 3</h1><br><br><br>
# %% [markdown]
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">This document provides a running example of completing the Week 3 assignment : </p>
# <ul class="simple">
# <li style="line-height:31px;">A shorter version with fewer comments is available as script: sparkMLlibClustering.py</li>
# <li style="line-height:31px;">To run these commands in Cloudera VM: first run the setup script: setupWeek3.sh</li>
# <li style="line-height:31px;">You can then copy paste these commands in pySpark. </li>
# <li style="line-height:31px;">To open pySpark, refer to : <a class="reference external" href="https://www.coursera.org/learn/machinelearningwithbigdata/supplement/GTFQ0/slides-module-2-lesson-3">Week 2</a> and <a class="reference external" href="https://www.coursera.org/learn/machinelearningwithbigdata/supplement/RH1zz/download-lesson-2-slides-spark-mllib-clustering">Week 4</a> of the Machine Learning course</li>
# <li style="line-height:31px;">Note that your dataset may be different from what is used here, so your results may not match with those shown here</li>
# </ul></div>

# %%
import pandas as pd
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array

# %% [markdown]
# <br><br>
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# <h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Step 1: Attribute Selection</h1>
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Import Data</h1><br><br> 
# 
# 
# <p style="line-height:31px;">First let us read the contents of the file ad-clicks.csv. The following commands read in the CSV file in a table format and removes any extra whitespaces. So, if the CSV contained ' userid  ' it becomes 'userid'. <br><br>
# 
# 
# Note that you must change the path to ad-clicks.csv to the location on your machine, if you want to run this command on your machine.
# </p>
# 
# </div>
# 
# <br><br><br><br>

# %%
adclicksDF = pd.read_csv('./ad-clicks.csv')
adclicksDF = adclicksDF.rename(columns=lambda x: x.strip()) #remove whitespaces from headers

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Let us display the first 5 lines of adclicksDF:</p>
# 
# </div>
# 
# <br><br><br><br>

# %%
adclicksDF.head(n=5)

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Next, We are going to add an extra column to the ad-clicks table and make it equal to 1. We do so to record the fact that each ROW is 1 ad-click. 
# You will see how this will become useful when we sum up this column to find how many ads
# did a user click.</p>
# 
# </div>
# 
# <br><br><br><br>

# %%
adclicksDF['adCount'] = 1

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Let us display the first 5 lines of adclicksDF and see if 
# a new column has been added:</p>
# 
# </div>
# 
# <br><br><br><br>

# %%
adclicksDF.head(n=5)

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Next, let us read the contents of the file buy-clicks.csv. As before, the following commands read in the CSV file in a table format and removes any extra whitespaces. So, if the CSV contained ' userid  ' it becomes 'userid'. <br><br>
# 
# 
# Note that you must change the path to buy-clicks.csv to the location on your machine, if you want to run this command on your machine.
# </p>
# 
# </div>
# 
# <br><br><br><br>

# %%
buyclicksDF = pd.read_csv('./buy-clicks.csv')
buyclicksDF = buyclicksDF.rename(columns=lambda x: x.strip()) #removes whitespaces from headers

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Let us display the first 5 lines of buyclicksDF:</p>
# 
# </div>
# 
# <br><br><br><br>

# %%
buyclicksDF.head(n=5)

# %% [markdown]
# <br><br>
# 
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Feature Selection</h1><br><br>
# 
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">For this exercise, we can choose from buyclicksDF,  the 'price' of each app that a user purchases as an attribute that captures user's purchasing behavior. The following command selects 'userid' and 'price' and drops all other columns that we do not want to use at this stage.</p>
# 
# 
# </div>
# 
# <br><br><br><br>

# %%
userPurchases = buyclicksDF[['userId','price']] #select only userid and price
userPurchases.head(n=5)

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Similarly, from the adclicksDF,  we will use the 'adCount' as an attribute that captures user's inclination to click on ads. The following command selects 'userid' and 'adCount' and drops all other columns that we do not want to use at this stage.</p>
# 
# 
# </div>
# 
# <br><br><br><br>

# %%
useradClicks = adclicksDF[['userId','adCount']]


# %%
useradClicks.head(n=5) #as we saw before, this line displays first five lines

# %% [markdown]
# <br><br>
# <h1 style="font-family: Arial; font-size:1.5em;color:#2462C0; font-style:bold">Step 2: Training Data Set Creation</h1>
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Create the first aggregate feature for clustering</h1><br><br> 
# 
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">From each of these single ad-clicks per row, we can now generate total ad clicks per user. Let's pick a user with userid = 3. To find out how many ads this user has clicked overall, we have to find each row that contains userid = 3, and report the total number of such rows.
# 
# The following commands sum the total number of ads per user and rename the columns to be called 'userid' and 'totalAdClicks'. <b> Note that you may not need to aggregate (e.g. sum over many rows) if you choose a different feature and your data set already provides the necessary information. </b> In the end, we want to get one row per user, if we are performing clustering over users.
# 
# </div>
# 
# <br><br><br><br>

# %%
adsPerUser = useradClicks.groupby('userId').sum()
adsPerUser = adsPerUser.reset_index()
adsPerUser.columns = ['userId', 'totalAdClicks'] #rename the columns

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Let us display the first 5 lines of 'adsPerUser' to see if there
# is a column named 'totalAdClicks' containing total adclicks per user.</p>
# 
# </div>
# 
# <br><br><br><br>

# %%
adsPerUser.head(n=5)

# %% [markdown]
# <br><br>
# 
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Create the second aggregate feature for clustering</h1><br><br> 
# 
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Similar to what we did for adclicks, here we find out how much money in total did each user spend on buying in-app purchases. As an example, let's pick a user with userid = 9. To find out the total money spent by this user, we have to find each row that contains userid = 9, and report the sum of the column'price' of each product they purchased.
# 
# The following commands sum the total money spent by each user and rename the columns to be called 'userid' and 'revenue'.
# <br><br>
# 
# <p style="line-height:31px;"> <b> Note: </b> that you can also use other aggregates, such as sum of money spent on a specific ad category by a user or on a set of ad categories by each user, game clicks per hour by each user etc. You are free to use any mathematical operations on the fields provided in the CSV files when creating features. </p>
# 
# 
# </div>
# 
# <br><br><br><br> 

# %%
revenuePerUser = userPurchases.groupby('userId').sum()
revenuePerUser = revenuePerUser.reset_index()
revenuePerUser.columns = ['userId', 'revenue'] #rename the columns


# %%
revenuePerUser.head(n=5)

# %% [markdown]
# <br><br>
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Merge the two tables</h1><br><br> 
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Lets see what we have so far. We have a table called revenuePerUser, where each row contains total money a user (with that 'userid') has spent. We also have another table called adsPerUser where each row contains total number of ads a user has clicked. We will use revenuePerUser and adsPerUser as features / attributes to capture our users' behavior.<br><br>
# 
# Let us combine these two attributes (features) so that each row contains both attributes per user. Let's merge these two tables to get one single table we can use for K-Means clustering.
# </div>
# 
# <br><br><br><br> 

# %%
combinedDF = adsPerUser.merge(revenuePerUser, on='userId') #userId, adCount, price

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# 
# <p style="line-height:31px;">Let us display the first 5 lines of the merged table. <b> Note: Depending on what attributes you choose, you may not need to merge tables. You may get all your attributes from a single table. </b></p>
# 
# </div>
# 
# <br><br><br><br>

# %%
combinedDF.head(n=5) #display how the merged table looks

# %% [markdown]
# <br><br>
# 
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Create the final training dataset</h1><br><br> 
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Our training data set is almost ready. At this stage we can remove the 'userid' from each row, since 'userid' is a computer generated random number assigned to each user. It does not capture any behavioral aspect of a user. One way to drop the 'userid', is to select the other two columns. </p>
# 
# </div>
# 
# <br><br><br><br>

# %%
trainingDF = combinedDF[['totalAdClicks','revenue']]
trainingDF.head(n=5)

# %% [markdown]
# <br><br>
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Display the dimensions of the training dataset</h1><br><br> 
# 
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Display the dimension of the training data set. To display the dimensions of the trainingDF, simply add .shape as a suffix and hit enter.</p>
# 
# </div>
# 
# <br><br><br><br>
# 

# %%
trainingDF.shape

# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">The following two commands convert the tables we created into a format that can be understood by the KMeans.train function. <br><br>
# 
# line[0] refers to the first column. line[1] refers to the second column. If you have more than 2 columns in your training table, modify this command by adding line[2], line[3], line[4] ...</p>
# 
# </div>
# 
# <br><br><br><br>
# 

# %%
sqlContext = SQLContext(sc)
pDF = sqlContext.createDataFrame(trainingDF)
parsedData = pDF.rdd.map(lambda line: array([line[0], line[1]])) #totalAdClicks, revenue

# %% [markdown]
# <br>
# <h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Step 3: Train to Create Cluster Centers</h1>
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Train KMeans model</h1><br><br> 
# %% [markdown]
# <br><br><br><br>
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Here we are creating two clusters as denoted in the second argument.</p>
# 
# </div>
# 
# <br><br><br><br>
# 

# %%
my_kmmodel = KMeans.train(parsedData, 2, maxIterations=10, runs=10, initializationMode="random")

# %% [markdown]
# <br><br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Display the centers of two clusters formed</h1><br><br> 

# %%
print(my_kmmodel.centers)

# %% [markdown]
# <br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Step 4: Recommend Actions</h1>
# <br><h1 style="font-family: Arial; font-size:1.5em;color:#2462C0">Analyze the cluster centers
# </h1>
# 
# <br><br> 
# 
# <div style="color:black;font-family: Arial; font-size:1.1em;line-height:65%">
# 
# <p style="line-height:31px;">Each array denotes the center for a cluster:<br><br>
# One Cluster is centered at   ... array([ 29.43211679,  24.21021898])<br>
# Other Cluster is centered at   ... array([  42.05442177,  113.02040816])</p>
# 
# <br><br>
# 
# <p style="line-height:31px;"> First number (field1) in each array refers to number of ad-clicks and the second number (field2) is the revenue per user.
# 
# Compare the 1st number of each cluster to see how differently users in each cluster behave when it comes to clicking ads.
# 
# Compare the 2nd number of each cluster to see how differently users in each cluster behave when it comes to buying stuff. 
# 
# </p><br><br>
# 
# <p style="line-height:31px;">In one cluster, in general, players click on ads much more often (~1.4 times) and spend more money (~4.7 times) on in-app purchases. Assuming that Eglence Inc. gets paid for showing ads and for hosting in-app purchase items, we can use this information to increase game's revenue by increasing the prices for ads we show to the frequent-clickers, and charge higher fees for hosting the in-app purchase items shown to the higher revenue generating buyers.</p>
# 
# <br><br>
# <p style="line-height:31px;"> <b> Note: </b>  This analysis requires you to compare the cluster centers and find any ‘significant’ differences in the corresponding feature values of the  centers. The answer to this question will depend on the features you have chosen. <br><br> Some features help distinguish the clusters remarkably while others may not tell you much. At this point, if you don’t find clear distinguishing patterns, perhaps re-running the clustering model with different numbers of clusters and revising the features you picked would be a good idea. </p>
# 
# </div>
# 

# %%




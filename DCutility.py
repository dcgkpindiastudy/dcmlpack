import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

def DC_describe_categorical(X):
    """
    Just like .decribe(), but returns the results for categorical variables only.
    """
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))

def DC_checkemail(email):
    
	# pass the regular expression
	# and the string into the fullmatch() method
	if(re.fullmatch(regex, email)):
		print("Valid Email")

	else:
		print("Invalid Email")

# def UVA_numeric(data, var_group):
# 	'''Univariate Analysis Numeric takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.
	
# 	Runs a loop: Calculates all the descriptive of i(th) variable and plot/print it'''

# 	size = len(var_group)
# 	plt.figure(figsize = (7*size, 3), dpi = 100)
# 	for j,i in enumerate(var_group):
# 		# calculating descriptives of variables
# 		mini = data[i].min()
# 		maxi = data[i].max()
# 		ran = data[i].max() - data[i].min()
# 		mean = data[i].mean()
# 		median = data[i].median()
# 		st_dev = data[i].std()
# 		skew = data[i].skew()
# 		kurt = data[i].kurtosis()

# 		# calculating the point of standard deviation 
# 		points = mean - st_dev, mean + st_dev

# 		# plotting the variables with every information
# 		plt.subplot(1, size, j+1)
# 		sns.kdeplot(data[i], shade = True, color = 'LightGreen')
# 		sns.lineplot(points, [0,0], color = 'black', label = "std_dev")
# 		sns.scatterplot([mini, maxi], [0,0], color = 'orange', label = "min/max")
# 		sns.scatterplot([mean], [0], color = 'red', label = "mean")
# 		sns.scatterplot([median], [0], color = 'blue', label = "median")
# 		plt.xlabel('{}'.format(i), fontsize = 20)
# 		plt.ylabel('density')
# 		plt.title("std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}".format((round(points[0],2),
# 																										round(points[1],2)),
# 																										round(kurt, 2),
# 																										round(skew, 2),
# 																										(round(mini, 2),
# 																										round(maxi, 2),
# 																										round(ran, 2)),
# 																										round(mean, 2),
# 																										round(median, 2)))

# def UVA_category(data, var_group):
# 	'''
# 	Univariate Analysis Categorical
# 	takes a group of variables (category) and plot/print all the value_counts and barplot.
# 	'''
# 	# setting figure size
# 	size = len(var_group)
# 	plt.figure(figsize=(8*size, 7), dpi = 100)

# 	# for every variable
# 	for j, i in enumerate(var_group):
# 		norm_count = data[i].value_counts(normalize = True)
# 		n_uni = data[i].nunique()

# 	# plotting the variable with every information

# 	plt.subplot(1, size, j+1)
# 	graph2 = sns.countplot(y=i, data = data, order = data[i].value_counts().index, palette="Set2")
# 	for p in graph2.patches:
# 		graph2.annotate(s='{:.0f}'.format(p.get_width()), xy=(p.get_width()+0.1, p.get_y()+0.7))
# 	plt.xlabel('fraction/percent', fontsize = 20)
# 	plt.ylabel('{}'.format(i), fontsize = 20)
# 	plt.title('n_uniques = {}\n value counts \n {};'.format(n_uni, norm_count))




def DC_cfplot(a, b, color, title):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import numpy as np
    cf_matrix = confusion_matrix(a, b)
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1,v2,v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.array(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap = color)
    ax.set_title(title+"\n\n");
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values');
    plt.show()
    
def DC_plot_roc_curve(fpr, tpr, label = None, rocscore = None):
    if rocscore is None:
        r = ""
    else:
        r = str(rocscore)
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(" AUC Plot ")
    plt.show()




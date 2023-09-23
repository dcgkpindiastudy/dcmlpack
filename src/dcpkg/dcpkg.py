import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'


def dcshape(X_train, X_test, y_train, y_test):
    print("The rows and column in X_train is :t", X_train.shape)
    print("The rows and column of X_test is:t", X_test.shape)
    print("The rows and column of y_train is:t", y_train.shape)
    print("The rows and column of y_test is:t", y_test.shape)
    

def dcsample(df, rw=10):
    import pandas as pd
    
    if rw is None:
        rw = 10
    
    return df.sample(rw)



def dcbar_plot_with_values(x_values, y_values, xlabel='', ylabel='', title=''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    
    # Create the bar plot
    bars = ax.bar(x_values, y_values)
    
    # Add values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()



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


def DC_cfplot(a, b, color, title):
    """
    Plot a confusion matrix heatmap with annotations.

    Parameters:
    a : array-like, shape (n_samples,)
        True binary labels.

    b : array-like, shape (n_samples,)
        Predicted binary labels.

    color : str or Colormap, optional
        Colormap for the heatmap.

    title : str
        Title for the confusion matrix plot.

    Returns:
    None
    """
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
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    fpr : array-like
        False Positive Rate values.

    tpr : array-like
        True Positive Rate values.

    label : str, optional
        Label for the ROC curve.

    rocscore : float, optional
        ROC AUC score to display in the plot.

    Returns:
    None
    """
    
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




# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:29:02 2020

@author: Jatin
"""



def DC_numericalCategoricalSplit(df):
    """
    Split a DataFrame into numerical and categorical data.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame to be split.

    Returns:
    numerical_data : pandas.DataFrame
        DataFrame containing only the numerical features.

    categorical_data : pandas.DataFrame
        DataFrame containing only the categorical features.

    Example:
    numerical_data, categorical_data = DC_numericalCategoricalSplit(my_dataframe)
    """
    numerical_features=df.select_dtypes(exclude=['object']).columns
    categorical_features=df.select_dtypes(include=['object']).columns
    numerical_data=df[numerical_features]
    categorical_data=df[categorical_features]
    return(numerical_data,categorical_data)



def DC_nullFind(df):
    """
    Find and count missing (null) values in a DataFrame.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame to check for missing values.

    Returns:
    null_numerical : pandas.Series
        Series containing counts of missing values for each numerical feature, sorted in descending order.

    null_categorical : pandas.Series
        Series containing counts of missing values for each categorical feature, sorted in descending order.

    Example:
    null_numerical, null_categorical = DC_nullFind(my_dataframe)
    """
    null_numerical=pd.isnull(df).sum().sort_values(ascending=False)
    #null_numerical=null_numerical[null_numerical>=0]
    null_categorical=pd.isnull(df).sum().sort_values(ascending=False)
    # null_categorical=null_categorical[null_categorical>=0]
    return(null_numerical,null_categorical)



def DC_removeNullRows(df,few_null_col_list):
    """
    Remove rows with missing (null) values in specified columns from a DataFrame.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame from which rows will be removed.
        
    few_null_col_list : list of str
        A list of column names. Rows with missing values in any of these columns will be removed.

    Returns:
    df_cleaned : pandas.DataFrame
        A new DataFrame with rows containing missing values in the specified columns removed.

    Example:
    cleaned_df = DC_removeNullRows(my_dataframe, ['column1', 'column2'])
    """
    for col in few_null_col_list:
        df=df[df[col].notnull()]
    return(df)
    
 


def DC_AutoEDA(df,labels,target_variable_name,
        data_summary_figsize=(16,16),corr_matrix_figsize=(16,16),
        data_summary_figcol="Reds_r",corr_matrix_figcol='Blues',
        corr_matrix_annot=False,
        pairplt_col='all',pairplt=False,
        feature_division_figsize=(12,12)):
    
    start_time = timeit.default_timer()
    
    #for converting class labels into integer values
    if df[target_variable_name].dtype=='object':
        class_labels=df[target_variable_name].unique().tolist()
        class_labels=[x for x in class_labels if type(x)==str]
        class_labels=[x for x in class_labels if str(x) != 'nan']
      
        for i in range(len(class_labels)):
            df[target_variable_name][df[target_variable_name]==class_labels[i]]=i
            
            
    df_orig=df
    print('The data looks like this: \n',df_orig.head())
    print('\nThe shape of data is: ',df_orig.shape)
    
    #To check missing values
    print('\nThe missing values in data are: \n',pd.isnull(df_orig).sum().sort_values(ascending=False))
    sns.heatmap(pd.isnull(df_orig))
    plt.title("Missing Values Summary", fontsize=(15), color="red")
    
    
   

    print('\nThe summary of data is: \n',df_orig.describe())
    plt.figure(figsize=data_summary_figsize)
    sns.heatmap(df_orig.describe()[1:].transpose(), annot= True, fmt=".1f",
                linecolor="black", linewidths=0.3,cmap=data_summary_figcol)
    plt.title("Data Summary", fontsize=(15), color="red")
    
      
   

    
    print('\nSome useful data information: \n')
    print(df_orig.info())
    print('\nThe columns in data are: \n',df_orig.columns.values)
    
    
    
   
    null_cutoff=0.5

    numerical=numericalCategoricalSplit(df_orig)[0]
    categorical=numericalCategoricalSplit(df_orig)[1]
    null_numerical=nullFind(numerical)[0]
    null_categorical=nullFind(categorical)[1]
    null=pd.concat([null_numerical,null_categorical])
    null_df=pd.DataFrame({'Null_in_Data':null}).sort_values(by=['Null_in_Data'],ascending=False)
    null_df_many=(null_df.loc[(null_df.Null_in_Data>null_cutoff*len(df_orig))])
    null_df_few=(null_df.loc[(null_df.Null_in_Data!=0)&(null_df.Null_in_Data<null_cutoff*len(df_orig))])

    many_null_col_list=null_df_many.index
    few_null_col_list=null_df_few.index
    
    #remove many null columns
    df_orig.drop(many_null_col_list,axis=1,inplace=True)
    
    df_wo_null=(removeNullRows(df_orig,few_null_col_list))
    
    
    if df_wo_null[target_variable_name].dtype=='object':
        df_wo_null[target_variable_name] =df_wo_null[target_variable_name].astype(str).astype(int)
   
    
    df=df_wo_null[df_wo_null.select_dtypes(exclude=['object']).columns]
   
    
    #Check correlation matrix
    plt.figure(figsize=corr_matrix_figsize)
    sns.heatmap(df.corr(),cmap=corr_matrix_figcol,annot=corr_matrix_annot) 
    
    
    col = df.columns.values
    number_of_columns=len(col)
    number_of_rows = len(col)-1/number_of_columns
    
    
    #To check Outliers
    plt.figure(figsize=(number_of_columns,number_of_rows))
    
    for i in range(0,len(col)):
        #plt.subplot(number_of_rows + 1,number_of_columns,i+1)
        if number_of_columns%2==0:
            plt.subplot(number_of_columns/2,2,i+1)   
            sns.set_style('whitegrid')
            sns.boxplot(df[col[i]],color='green',orient='h')
            plt.tight_layout()
        else:
            plt.subplot((number_of_columns+1)/2,2,i+1)
            sns.set_style('whitegrid')
            sns.boxplot(df[col[i]],color='green',orient='h')
            plt.tight_layout()
    
    
    #To check distribution-Skewness
    for i in range(0,len(col)):
        fig,axis = plt.subplots(1, 2,figsize=(16, 5))
        sns.distplot(df_orig[col[i]],kde=True,ax=axis[0]) 
        axis[0].axvline(df_orig[col[i]].mean(),color = "k",linestyle="dashed",label="MEAN")
        axis[0].legend(loc="upper right")
        axis[0].set_title('distribution of {}. Skewness = {:.4f}'.format(col[i] ,df_orig[col[i]].skew()))
        
        sns.violinplot(x=target_variable_name, y=col[i], data=df_orig, ax=axis[1], inner='quartile')
        axis[1].set_title('violin of {}, split by target'.format(col[i]))
    
       
    
    #to construct pairplot
    if (pairplt==True) and (pairplt_col!='all'):
        sns.pairplot(data=df, vars=pairplt_col, hue=target_variable_name)
    elif (pairplt==True) and (pairplt_col=='all'):
        sns.pairplot(data=df, vars=df.columns.values, hue=target_variable_name)
   
    
    
    #Proportion of target variable in dataset   
    
    st=df[target_variable_name].value_counts().sort_index()
    print('\nThe target variable is divided into: \n',st) #how many belong to each class of target variable
    
    
    
    plt.figure(figsize=feature_division_figsize)
    plt.subplot(121)
    ax = sns.countplot(y = df_orig[target_variable_name],
                     
                       linewidth=1,
                       edgecolor="k"*2)
    for i,j in enumerate(st):
        ax.text(.7,i,j,weight = "bold",fontsize = 27)
    plt.title("Count for target variable in datset")
    
    
    plt.subplot(122)
    plt.pie(st,
            labels=labels,
            autopct="%.2f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
    my_circ = plt.Circle((0,0),.7,color = "white")
    plt.gca().add_artist(my_circ)
    plt.subplots_adjust(wspace = .2)
    plt.title("Proportion of target variable in dataset")
    
    
    print('\nThe numerical features are: \n',df_wo_null.select_dtypes(exclude=['object']).columns.tolist())
    print('\nThe categorical features are: \n',df_wo_null.select_dtypes(include=['object']).columns.tolist())
    
    #Proportion of categorical variables in dataset   
    if len(df_wo_null.select_dtypes(include=['object']).columns.tolist())>=1:
        for cat_feat in df_wo_null.select_dtypes(include=['object']).columns.tolist():
            
            ct=df_wo_null.select_dtypes(include=['object'])[cat_feat].value_counts().sort_values(ascending=False)
            print('\nThe categorical variable is divided into: \n',ct) #how many belong to each class of target variable
            
            
            if (ct.index.size)<50:
                plt.figure(figsize=feature_division_figsize)
                plt.subplot(121)
                ax = sns.countplot(y = df_wo_null.select_dtypes(include=['object'])[cat_feat],
                                  
                                   linewidth=1,
                                   edgecolor="k"*2)
                for i,j in enumerate(ct):
                    ax.text(.7,i,j,weight = "bold",fontsize = 27)
                plt.title("Count for categorical variable in datset")
                
                
                plt.subplot(122)
                plt.pie(ct,
                        labels=df_wo_null.select_dtypes(include=['object'])[cat_feat].unique().tolist(),
                        autopct="%.2f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
                my_circ = plt.Circle((0,0),.7,color = "white")
                plt.gca().add_artist(my_circ)
                plt.subplots_adjust(wspace = .2)
                plt.title("Proportion of categorical variable in dataset")
            else:
                print('\nThe categorical variable %s has too many divisions to plot \n'%cat_feat)
            continue
    elapsed = timeit.default_timer() - start_time
    print('\nExecution Time for EDA: %.2f minutes'%(elapsed/60))
    
    
    return df_wo_null,df_wo_null.select_dtypes(exclude=['object']).columns.tolist(),df_wo_null.select_dtypes(include=['object']).columns.tolist()




def dc_calculate_months_difference(x, startdate):
    """
    Calculate the difference in months between a given date and a start date.

    Parameters:
    x : datetime
        The target date for which you want to calculate the difference.

    startdate : str
        The start date as a string in the format 'YYYY-mm-dd'.

    Returns:
    months_diff : int or None
        The difference in months between the two dates. Returns None if there's an exception.

    Example:
    date_difference = dc_calculate_months_difference(datetime(2023, 9, 15), '2023-01-01')
    """
    from dateutil.relativedelta import relativedelta
    d = startdate
    try:
        diff = relativedelta(pd.to_datetime(d), x)
        months_diff = diff.years * 12 + diff.months
        return months_diff
    except Exception as e:
        return None  # Handle cases where the date conversion or calculation fails

def dcvaluecountbarh(d):
    """
    Create a horizontal bar plot of value counts for a given variable from a dataframe.

    Parameters:
    d : pandas Series
        The variable (column) from your dataframe for which you want to create a value count plot.

    Example:
    dcvaluecountbarh(df["XYZ"])
    
    This function will generate a horizontal bar plot displaying the count of each unique value in the specified variable.
    """
    import matplotlib.pyplot as plt  
    ax = d.value_counts().plot(kind='barh')
    # Annotate each bar with its value
    for p in ax.patches:
        ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')

    # Show the plot
    plt.show()
    

# WoE function for discrete unordered variables
def dc_woe_discrete(df, discrete_variabe_name, good_bad_variable_df):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a discrete variable.

    Args:
        df (DataFrame): The entire DataFrame containing both the discrete variable and the dependent variable.
        discrete_variable_name (str): The name of the independent discrete variable in the DataFrame.
        good_bad_variable_df (DataFrame): The dependent variable (good/bad) in the form of a DataFrame.

    Returns:
        DataFrame: Returns a DataFrame with WoE, IV, and other details for the discrete variable.

    Example:
    df_woe_tab = dc_woe_discrete(df, 'DiscreteVariableName', df['GoodBadColumn'])
    This function calculates the WoE and IV for a discrete variable and returns a DataFrame with the results.
    """
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
# Here we combine all of the operations above in a function.
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.


def dc_plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    """
    Plot Weight of Evidence (WoE) for a DataFrame.

    Parameters:
    - df_WoE (DataFrame): The input DataFrame containing WoE values.
    - rotation_of_x_axis_labels (int, optional): Rotation angle for x-axis labels (default: 0).

    This function plots the Weight of Evidence (WoE) values from a DataFrame and annotates the values
    on the plot. It is useful for visualizing the relationship between a variable and its WoE values.

    Example:
    >>> df = pd.DataFrame({'Category': ['A', 'B', 'C', 'D'], 'WoE': [0.1, -0.2, 0.3, -0.4]})
    >>> plot_by_woe(df, rotation_of_x_axis_labels=45)
    
    """
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    # Turns the values of the column with index 0 to strings, makes an array from these strings, and passes it to variable x.
    y = df_WoE['WoE']
    # Selects a column with label 'WoE' and passes it to variable y.
    
    plt.figure(figsize=(18, 6))
    # Sets the graph size to width 18 x height 6.
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    # Plots the datapoints with coordiantes variable x on the x-axis and variable y on the y-axis.
    # Sets the marker for each datapoint to a circle, the style line between the points to dashed, and the color to black.
    plt.xlabel(df_WoE.columns[0])
    # Names the x-axis with the name of the column with index 0.
    plt.ylabel('Weight of Evidence')
    # Names the y-axis 'Weight of Evidence'.
    plt.title('Weight of Evidence by ' + df_WoE.columns[0])
    # Names the grapth 'Weight of Evidence by ' the name of the column with index 0.
    
    for i, j in zip(x, y):
        plt.annotate(f'{j:.2f}', (i, j), textcoords='offset points', xytext=(0, 10), ha='center')
    # put the values of weight of evidence against each data coordinates
    
    plt.xticks(rotation=rotation_of_x_axis_labels)
    # Rotates the labels of the x-axis a predefined number of degrees.
    plt.show()
    
# WoE function for ordered discrete and continuous variables
def dc_woe_ordered_continuous(df, OC_variabe_name, good_bad_variable_df):
    """Calculate Weight of Evidence (WoE) for ordered discrete and continuous variables.

    Parameters:
    - df (DataFrame): The input DataFrame containing the variable of interest and a good/bad indicator.
    - OC_variabe_name (str): The name of the ordered discrete or continuous variable.
    - good_bad_variable_df (DataFrame): A DataFrame containing the good/bad indicator variable or target variable.

    Returns:
    - DataFrame: A DataFrame containing WoE, IV, and other statistics for the variable.

    This function calculates WoE and Information Value (IV) for an ordered discrete or continuous variable.
    It groups the data by the specified variable, calculates various statistics, and returns the results.

    Example:
    >>> df = pd.DataFrame({'Age': [25, 35, 45, 30, 40],
    ...                    'Target': [0, 1, 0, 1, 0]})
    >>> good_bad_variable_df = df[['Target']]
    >>> result = woe_ordered_continuous(df, 'Age', good_bad_variable_df)
    >>> print(result)
    """
    df = pd.concat([df[OC_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
# Here we define a function similar to the one above, ...
# ... with one slight difference: we order the results by the values of a different column.
# The function takes 3 arguments: a dataframe, a string, and a dataframe. The function returns a dataframe as a result.




class DC_LogisticRegression_with_p_values:
    """
     A custom logistic regression model with p-value calculation for coefficient significance.

    This class extends the functionality of scikit-learn's LogisticRegression model to calculate
    p-values for each coefficient in the logistic regression model.

    Parameters:
    *args, **kwargs: Additional arguments and keyword arguments passed to the LogisticRegression model.

    Attributes:
    model: LogisticRegression
        The scikit-learn LogisticRegression model fitted to the data.
    coef_: array, shape (1, n_features)
        Coefficients of the logistic regression model.
    intercept_: array, shape (1,)
        Intercept (bias) of the logistic regression model.
    p_values: list
        Two-tailed p-values corresponding to the significance of each coefficient in the model.

    Methods:
    fit(X, y):
        Fit the logistic regression model to the input data.

    Note:
    The p-values are calculated using the Cramer-Rao lower bound method to estimate the standard errors
    of the coefficients and then computing z-scores for hypothesis testing.

    Reference:
    https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound
    
    Example:
    
    reg2 = LogisticRegression_with_p_values()
    reg2.fit(inputs_train, loan_data_targets_train)
    
    feature_name = inputs_train.columns.values
    
    # Summary data.
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg2.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg2.intercept_[0]]
    summary_table = summary_table.sort_index()
    summary_table
    
    # We add the 'p_values' here, just as we did before.
    p_values = reg2.p_values
    p_values = np.append(np.nan,np.array(p_values))
    summary_table['p_values'] = p_values
    summary_table
    # Here we get the results for our final PD model.
    
    
    """
    from sklearn import linear_model
    import scipy.stats as stat
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X.astype(np.float64) / denom).T, X.astype(np.float64))  # Ensure X is of dtype float64
        Cramer_Rao = np.linalg.inv(F_ij)
        # F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        # Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values
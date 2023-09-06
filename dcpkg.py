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


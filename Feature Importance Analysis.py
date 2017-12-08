
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from numpy import logspace
import pickle
from scipy.stats import rankdata

import shap

from sklearn.ensemble import RandomForestRegressor
import TraitData
from imp import reload
import seaborn as sns
reload(TraitData)
sns.set()
sns.set(font_scale=1.5)

# try:
#     SAVED_STATE = pickle.load(open("np.random.state", 'rb'))
#     np.random.set_state(SAVED_STATE)
# except Exception as e:
#     SAVED_STATE = False
    

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (7,15)
get_ipython().magic("config InlineBackend.figure_format = 'retina'")


# <img style="float:right" src="https://www.washington.edu/brand/files/2014/09/W-Logo_Purple_Hex.png" width=60px)/>
# <h1> Feature Interpretation Across Methods: A Study <small> Tony Cannistra | CSE546 Au17 Final Project </small></h1>
# We are attempting to understand the type and magnitude of the effects that various physiological or behavioral traits have had on historical range shifts.
# 
# The challenge is that standard linear regression has been only able to explain a small percentage of the variance seen in range shift data for several taxa ([Angert et al., 2011](http://onlinelibrary.wiley.com/doi/10.1111/j.1461-0248.2011.01620.x/abstract)). This is likely as a result of the relationships between predictor and response variables being something other than linear, especially since several variables have biological relevance for their ability to facilitate range shifts. 

# ## Load Data + Define Functions
# For this exploration we have trait data for several large taxonomic groups of organisms and their observed range shifts over the past 100 years. There are 3 datasets
# ### Define Each Dataset

# In[2]:

plantsData = {
    'file' : "../data/angert_etal_2011/plants5.csv",
    
    'responseVar'   : "migration_m",

    'drop_features' : ["Taxon",
                     "migr_sterr_m", 
                     "shift + 2SE", 
                     'signif_shift',
                     "signif_shift2",
                     "dispmode01",
                     "DispModeEng", ## what is this
                     "shift + 2SE", 
                      "Grime"],
    'categorical_features' : ["oceanity",
                            "dispersal_mode",
                            "BreedSysCode"]
}

mammalData = {
    'file' : "../data/angert_etal_2011/mammals01.csv",
    
    'responseVar'   : "High_change",

    'drop_features' : ["Taxon",
                       "High_change_pfa2",
                       "Daily_rhythm_code",
                       "Annual_rhythm_code"],
    
    'categorical_features' : ["Daily01",
                              "Annual01",
                              "Food01",
                              "Daily_rhythm",
                              "Annual_rhythm",
                              "Food"]
}

rankings = []
MSEs = []


# ### Select, Load, and Normalize Dataset

# In[3]:

dataset = mammalData
dropNA  = 0

td = TraitData.TraitData(dataset['file'],
                         dataset['responseVar'],
                         dataset['drop_features'],
                         dataset['categorical_features'],
                         dropNA=dropNA, scale=True)

print(len(td.X))
display(td.X.head())
td.X = np.array(td.X)

print(td.X.shape)
print(td.feature_names)



# In[4]:

pickle.dump(open('np.random.state', 'wb'), np.random.save_state())
X_train, X_test, Y_train, Y_test = train_test_split(td.X, td.Y, train_size=0.60)


# ## Learning

# In[5]:

## Importance Plotting Function
def plot_importance(importances, names, title=None):
    f, ax = plt.subplots()
    bar_indices = np.arange(len(names))
    bar_width = 0.45
    importances_ind = np.argsort(abs(importances))

    plt.barh(bar_indices,
             importances[importances_ind],
            align='center', color='#4b2e83')


    plt.yticks(bar_indices, np.array(names)[importances_ind])
    plt.tight_layout()
    plt.title(title)
    plt.show()
    
def compute_ranks(importances, absolute=True):
    if absolute:
        sorted_idx =  abs(importances).argsort()[::-1]
    else:
        sorted_idx = importances.argsort()[::-1]
    return(np.arange(len(importances))[sorted_idx.argsort()])


# ### Linear Models
# #### OLS

# In[6]:


ols = LinearRegression(fit_intercept=False, normalize=False)
ols.fit(X_train, Y_train)
preds = ols.predict(X_test)
print(preds, Y_test)
error = mean_squared_error(Y_test, preds)
print(error)
plot_importance(ols.coef_, td.feature_names, "OLS (MSE: %1.4f)" % error)
MSEs.append(("OLS", error))
rankings.append(['OLS'] + list(compute_ranks(ols.coef_)))


# #### Ridge CV

# In[7]:

ridgecv = RidgeCV(normalize=False)
ridgecv.fit(X_train, Y_train)
preds = ridgecv.predict(X_test)
error = mean_squared_error(Y_test, preds)
MSEs.append(("Ridge", error))

plot_importance(ridgecv.coef_, td.feature_names,  "Ridge CV (MSE: %1.4f)" % error)
rankings.append(['Ridge'] + list(compute_ranks(ridgecv.coef_)))


# #### Kernel Ridge

# In[8]:

kr = KernelRidge(kernel='rbf')
kr.fit(X_train, Y_train)
preds = kr.predict(X_test)
error = mean_squared_error(Y_test, preds)
#plot_importance(kr.coef_, td.feature_names,  "Lasso CV (MSE: %1.4f)" % error)
shapdata = shap.DenseData(X_train, td.feature_names)
explainer = shap.KernelExplainer(kr.predict, shapdata, nsamples=100)
explanations =[]
for i in range(0, len(X_test)-1):
    try: 
        explanations.append(explainer.explain(np.mat(X_test[i:i+1, :])))
    except Exception as e:
        print("error on ", (i, i+1))
        print(e)
        continue

importances = np.mean(np.array([exp.effects for exp in explanations]), axis=0)
plot_importance(importances, td.feature_names, "Kernel Ridge (MSE: %1.4f)" % error)
MSEs.append(("Kernel Ridge", error))

rankings.append(['Kernel Ridge'] + list(compute_ranks(importances)))


# ### Trees

# In[9]:

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
preds = rf.predict(X_test)
error = mean_squared_error(Y_test, preds)
MSEs.append(("RF", error))

plot_importance(rf.feature_importances_, td.feature_names, "RF (MSE: %1.4f)" %error)
rankings.append(['RF'] + list(compute_ranks(rf.feature_importances_)))


# ### Support Vectors

# In[10]:

baseModel = SVR()
params_grid = {
    'C'     : np.logspace(-3, 1.2),
    'gamma' : np.logspace(-3, 1.2),
    'degree': [0, 1, 2],
    'kernel': ['linear', 'poly', 'rbf']
}
gridSearch = GridSearchCV(baseModel,
                         param_grid = params_grid,
                         scoring="neg_mean_squared_error",
                         error_score = 0,
                         n_jobs = -1)
gridSearch.fit(X_train, Y_train)


# In[11]:

best = gridSearch.best_estimator_


# In[12]:

preds = gridSearch.best_estimator_.predict(X_test)
error = mean_squared_error(preds, Y_test)
print(error)
MSEs.append(("SVR", error))


# In[13]:

shap.initjs()
shapdata = shap.DenseData(X_train, td.feature_names)
explainer = shap.KernelExplainer(best.predict, shapdata, nsamples=100)
explanations =[]
for i in range(0, len(X_test)-1):
    try: 
        explanations.append(explainer.explain(np.mat(X_test[i:i+1, :])))
    except Exception as e:
        print("error on ", (i, i+1))
        print(e)
        continue

all_effects = pd.concat([pd.DataFrame(list(zip(exp.effects, td.feature_names))) for exp in explanations])


# In[ ]:




# In[14]:

importances = np.mean(np.array([exp.effects for exp in explanations]), axis=0)
plot_importance(importances, td.feature_names, "Grid SVR (MSE: %1.4f)" % error)
rankings.append(['SVR'] + list(compute_ranks(importances)))


# In[15]:

results = pd.DataFrame(rankings, columns=["Method"] + td.feature_names).set_index("Method").T
results['mean'] = results.mean(axis=1)


# In[16]:

results.sort_values(by=['mean'])


# In[17]:

from pandas.tools.plotting import table


# In[18]:

results = results.sort_values(by='mean')
fig = plt.figure(figsize=(7, 6))
ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
tab = table(ax, results, loc='upper left')
tab.auto_set_font_size(False)
tab.set_fontsize(14)
tab.scale(1.2, 1.3)
fig.tight_layout()


# In[19]:


fig = plt.figure(figsize=(8, 4))
labels, errors = zip(*MSEs)
errors = np.array(errors)
labels = np.array(labels)
sorted_idx = np.argsort(errors)[::-1]
plt.bar(range(0, len(MSEs)), errors[sorted_idx])
plt.xticks(range(0, len(MSEs)), labels[sorted_idx])
plt.ylabel("MSE")
plt.title("Model Performance Comparison")


# 

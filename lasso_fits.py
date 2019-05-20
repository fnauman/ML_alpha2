# Part of the code follows Jeremy Howard's fast.ai course Machine learning
# http://course18.fast.ai/ml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lasso_imp(mod,df):
    return pd.DataFrame({'Features':df.columns, 'Importance':mod.coef_}).sort_values('Importance', ascending=False)

def plot_las_imp(l_imp): 
    return l_imp.plot('Features', 'Importance', 'barh', figsize=(12,7), legend=False)

def lasso_gridcv(df, fld='Ex', pth='', name=None, fi_plts=True,
                 test_size=None, zave=False):
    '''
    Grid search with cross validation
    Training and test splits are not random
    '''

    from preprocess import train_test_seq
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error #mean_squared_error
    
    tst_sz = test_size if test_size is not None else 0.2
    totsz = df.shape[0]  
    train_size = 1 - tst_sz

    title  = name if name is not None else 'df'

    if zave:
        flds = ['Ex2l','Ey2l']
    else:
        flds = ['Ex','Ey']

    X_train, X_test, y_train, y_test = train_test_seq(df.drop(flds,axis=1),df[fld],test_size=tst_sz)

    print("Test,train shapes:",X_test.shape,X_train.shape)
    
    # GRID SEARCH CV
    
    # Setup the pipeline steps: steps
    steps = [('lasso', Lasso(fit_intercept=False))]

    # Create the pipeline: pipeline 
    pipeline = Pipeline(steps)

    # Specify the hyperparameter space
    parameters = {'lasso__alpha': [1e-3,1e-2,1e-1,5e-1,1.0]}

    # Use TimeSeriesSplit instead of the default random splits used by GridSearchCV
    my_cv = TimeSeriesSplit(n_splits=2).split(X_train)
    
    # Create the GridSearchCV object: gm_cv
    # NOTE: return_train_score=True is necessary for train/test scores vs alpha
    gm_cv = GridSearchCV(pipeline,parameters,cv=my_cv,verbose=True,n_jobs=4,scoring='neg_mean_absolute_error',return_train_score=True,iid=False)

    # Fit to the training set
    gm_cv.fit(X_train,y_train)

    # Predict
    y_pred = gm_cv.predict(X_test)
    

    
    # https://github.com/amueller/COMS4995-s19/blob/master/slides/aml-06-linear-models-regression/aml-05-linear-models-regression.ipynb
    means = gm_cv.cv_results_['mean_test_score']
    stds  = gm_cv.cv_results_['std_test_score']
    print(f"Means of CV folds: {means}")
    print(f"STDs of CV folds : {stds}")

    results = pd.DataFrame(gm_cv.cv_results_)
    print(results.head())
    results.plot('param_lasso__alpha', 'mean_train_score')
    results.plot('param_lasso__alpha', 'mean_test_score', ax=plt.gca())
    plt.fill_between(results.param_lasso__alpha.astype(np.float),
                 results['mean_train_score'] + results['std_train_score'],
                 results['mean_train_score'] - results['std_train_score'], alpha=0.2)
    plt.fill_between(results.param_lasso__alpha.astype(np.float),
                 results['mean_test_score'] + results['std_test_score'],
                 results['mean_test_score'] - results['std_test_score'], alpha=0.2)
    plt.legend()
    plt.xscale("log")
    
    plt.savefig("lasso_grid_score_vs_alpha.pdf", bbox_inches='tight')

    
    
    
    
    # Compute and print the metrics
    print(f"Tuned Lasso Alpha: {gm_cv.best_params_}")
    print(f"Tuned Lasso Score: {gm_cv.score(X_test, y_test)}")
  
    # FIT model to tuned parameters
    print('BEST FIT model')
    lasso_mod = Lasso(alpha=float(gm_cv.best_params_['lasso__alpha']),fit_intercept=False)
    lasso_mod.fit(X_train,y_train)
    y_pred = lasso_mod.predict(X_test)
    
    # BEST FIT model - save
    from joblib import dump, load
    dump(lasso_mod, 'bst_lasso.joblib')
    # LOAD like this:
    # bst_lasso = load('bst_lasso.joblib')
    
    # PLOT importances
    l_imp = lasso_imp(lasso_mod,df.drop(flds,axis=1))

    if fi_plts:
        fiplt = l_imp.plot('Features', 'Importance', 'barh', figsize=(12,7), legend=False)
        fig   = fiplt.get_figure()
        fig.savefig(pth + str(title) + '_fi_las.pdf',bbox_inches='tight')
    else:
        plot_las_imp(l_imp)

    print(f"Mean Absolute Error: {mean_absolute_error(y_pred,y_test)}")
    print(f"LASSO score: {lasso_mod.score(X_test,y_test)}")
    print(f"Coefficients: {lasso_mod.coef_}")

    return y_train,y_pred,y_test

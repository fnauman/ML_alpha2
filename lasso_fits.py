# Part of the code follows Jeremy Howard's fast.ai course Machine learning
# http://course18.fast.ai/ml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lasso_imp(mod,df):
    return pd.DataFrame({'Features':df.columns, 'Importance':mod.coef_}).sort_values('Importance', ascending=False)

def plot_las_imp(l_imp): 
    return l_imp.plot('Features', 'Importance', 'barh', figsize=(12,7), legend=False)

def lasso_gridcv(df, fld='Ex', pth='linear_lasso/', name=None, fi_plts=True,
                 test_size=None, newmodel=False, zave=False):
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
    steps = [('lasso', Lasso())]

    # Create the pipeline: pipeline 
    pipeline = Pipeline(steps)

    # Specify the hyperparameter space
    parameters = {'lasso__alpha': [1e-3,1e-2,1e-1,5e-1,1.0]}

    # Use TimeSeriesSplit instead of the default random splits used by GridSearchCV
    my_cv = TimeSeriesSplit(n_splits=2).split(X_train)
    
    # Create the GridSearchCV object: gm_cv
    gm_cv = GridSearchCV(pipeline,parameters,cv=my_cv,verbose=True,n_jobs=4,scoring='neg_mean_absolute_error')

    # Fit to the training set
    gm_cv.fit(X_train,y_train)

    # Predict
    y_pred = gm_cv.predict(X_test)

    # Compute and print the metrics
    print(f"Tuned Lasso Alpha: {gm_cv.best_params_}")
    print(f"Tuned Lasso Score: {gm_cv.score(X_test, y_test)}")
  
    # FIT model to tuned parameters
    print('BEST FIT model')
    lasso_mod = Lasso(alpha=float(gm_cv.best_params_['lasso__alpha']))
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

    if newmodel:
        print('NEW model: Dropping unimportant features')
  
        # Drop un-important features
        #print('Importances: ',l_imp)

        to_keep = l_imp.iloc[l_imp['Importance'].abs().argsort()[::-1]].head(2)
        df_keep = df[to_keep['Features']].copy()
  
        X_train, X_test, y_train, y_test = train_test_seq(df_keep,df[fld],test_size=tst_sz)

        lasso_mod.fit(X_train,y_train)
        y_pred = lasso_mod.predict(X_test)

        print(f"Mean Absolute Error: {mean_absolute_error(y_pred,y_test)}")
        print(lasso_mod.score(X_test,y_test))
        print(f"Coefficients: {lasso_mod.coef_}")
  
        # PLOT importances
        l_imp = lasso_imp(lasso_mod,df_keep)
        plot_las_imp(l_imp)
        
    return y_train,y_pred,y_test

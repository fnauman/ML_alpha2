# Part of the code follows Jeremy Howard's fast.ai course Machine learning
# http://course18.fast.ai/ml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rf_imp(mod,df):
    # only used for SKLEARN's feature importances
    return pd.DataFrame({'Features': df.columns, 'Importance': mod.feature_importances_}).sort_values('Importance', ascending=False)

def plot_rf_imp(rf_imp): 
    return rf_imp.plot('Features', 'Importance', 'barh', figsize=(12,7), legend=False, colormap='RdBu')

def rf_gridcv(df, fld='Ex', pth='linear/', name=None, fi_plts=False,                           test_size=None, newmodel=False, zave=False, err_metric='mae'):
    '''
    Grid search with cross validation
    Training and test splits are not random
    '''

    from preprocess import train_test_seq
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    tst_sz = test_size if test_size is not None else 0.2
    totsz = df.shape[0]  
    train_size = 1 - tst_sz

    title  = name if name is not None else 'df'
    
    if zave:
        flds = ['Ex2l','Ey2l']
        # Specify the hyperparameter space
        parameters = {'rf__max_depth': [4,8],
                      'rf__max_features': ['auto', 'sqrt', None],
                      'rf__min_samples_leaf': [4, 8, 16],
                      'rf__n_estimators': [8,16,32,64]}

    else:
        flds = ['Ex','Ey']
        # Specify the hyperparameter space
        parameters = {'rf__max_depth': [2,4,8],
                      'rf__max_features': ['auto', 'sqrt', None],
                      'rf__min_samples_leaf': [2, 4, 8],
                      'rf__n_estimators': [4,8,16,32]}
        
    if err_metric=='mae':
        scoring = 'neg_mean_absolute_error'
    else:
        scoring = 'neg_mean_squared_error'

    
    X_train, X_test, y_train, y_test = train_test_seq(df.drop(flds,axis=1),
                                                   df[fld],test_size=tst_sz)

    print("Test,train shapes:",X_test.shape,X_train.shape)
    
    # GRID SEARCH CV
    
    # Setup the pipeline steps: steps
    steps = [('rf', RandomForestRegressor(criterion = err_metric, bootstrap = False, random_state = 42))] 

    # Create the pipeline: pipeline 
    pipeline = Pipeline(steps)

    # Use TimeSeriesSplit instead of the default random splits used by GridSearchCV
    my_cv = TimeSeriesSplit(n_splits=2).split(X_train)

    # Create the GridSearchCV object: gm_cv
    gm_cv = GridSearchCV(pipeline, parameters, cv=my_cv, verbose=True, n_jobs=4, scoring=scoring)

    # Fit to the training set
    gm_cv.fit(X_train,y_train)

    # Predict
    y_pred = gm_cv.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test,y_pred)}")

    # Compute and print the metrics
    print(f"Tuned RF params: {gm_cv.best_params_}")
    print(f"Tuned RF score:  {gm_cv.score(X_test, y_test)}")
  
    # FIT model to tuned parameters
    rf_mod = RandomForestRegressor(criterion=err_metric,
             bootstrap=False,
             max_depth=gm_cv.best_params_['rf__max_depth'],
             max_features=gm_cv.best_params_['rf__max_features'],
             min_samples_leaf=gm_cv.best_params_['rf__min_samples_leaf'],
             n_estimators=gm_cv.best_params_['rf__n_estimators'],
             random_state = 42) # fix random state for reproducibility
    rf_mod.fit(X_train,y_train)
    y_pred = rf_mod.predict(X_test)
    
    # BEST FIT model - save
    from joblib import dump, load
    dump(rf_mod, 'bst_rf.joblib')
    # LOAD like this:    
    # bst_rf = load('bst_rf.joblib')

    print(f"MAE: {mean_absolute_error(y_pred,y_test)}")
    print(f"R^2: {rf_mod.score(X_test,y_test)}")

    # PLOT importances
    print('Best model important features: SKLEARN')
    print(rf_mod.feature_importances_)
    fi = rf_imp(rf_mod,df.drop(flds,axis=1))
    plot_rf_imp(fi)
    
    if fi_plts:
        fiplt = fi.plot('Features', 'Importance', 'barh', figsize=(12,7), legend=False)
        fig   = fiplt.get_figure()
        fig.savefig(pth + str(title) + '_fi_skl.pdf',bbox_inches='tight')

    if newmodel:
        print('NEW model: Dropping unimportant features')
        to_keep = fi.iloc[fi['Importance'].abs().argsort()[::-1]].head(2)
        df_keep = df[to_keep.cols].copy()
  
        X_train, X_test, y_train, y_test = train_test_seq(df_keep, df[fld], test_size=tst_sz)

        rf_mod.fit(X_train,y_train)
        y_pred = rf_mod.predict(X_test)

        print(f"MAE: {mean_absolute_error(y_pred,y_test)}") 
        print(f"R^2: {rf_mod.score(X_test,y_test)}")
    
    return y_train,y_pred,y_test#,rf_mod

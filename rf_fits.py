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

def rf_gridcv(df, fld='Ex', pth='', name=None, fi_plts=False, test_size=None, newmodel=False, zave=False, err_metric='mae'):
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
    gm_cv = GridSearchCV(pipeline, parameters, cv=my_cv, verbose=True, n_jobs=4, scoring=scoring, return_train_score=True)

    # Fit to the training set
    gm_cv.fit(X_train,y_train)

    
    
    # Print SCORES with deviations
    means = gm_cv.cv_results_['mean_test_score']
    stds  = gm_cv.cv_results_['std_test_score']
    print(f"Means of CV folds: {means}")
    print(f"STDs of CV folds : {stds}")

    # https://github.com/amueller/COMS4995-s19/blob/master/slides/aml-08-trees-forests/aml-10.ipynb
    # Plot error vs various hyperparameters
    scores = pd.DataFrame(gm_cv.cv_results_)
    print(scores.head())
    
    plt.figure(0)
    
    scores.plot('param_rf__max_depth', 'mean_train_score')
    scores.plot('param_rf__max_depth', 'mean_test_score', ax=plt.gca())
    plt.fill_between(scores.param_rf__max_depth.astype(np.float),
                 scores['mean_train_score'] + scores['std_train_score'],
                 scores['mean_train_score'] - scores['std_train_score'], alpha=0.2)
    plt.fill_between(scores.param_rf__max_depth.astype(np.float),
                 scores['mean_test_score'] + scores['std_test_score'],
                 scores['mean_test_score'] - scores['std_test_score'], alpha=0.2)
    plt.legend()
    plt.savefig("rf_grid_max_depth.pdf", bbox_inches="tight")
    
    
    plt.figure(1)
    scores.plot(x='param_rf__max_depth', y='mean_train_score', yerr='std_train_score', ax=plt.gca())
    scores.plot(x='param_rf__max_depth', y='mean_test_score', yerr='std_test_score', ax=plt.gca())
    plt.savefig("rf_grid_max_depth_.pdf", bbox_inches="tight")    
    
    # Plot error vs various hyperparameters
    plt.figure(2)
    scores.plot(x='param_rf__n_estimators', y='mean_train_score', yerr='std_train_score', ax=plt.gca())
    scores.plot(x='param_rf__n_estimators', y='mean_test_score', yerr='std_test_score', ax=plt.gca())
    plt.savefig("rf_grid_n_estimators.pdf", bbox_inches="tight")    
    
   
    
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

    return y_train,y_pred,y_test#,rf_mod

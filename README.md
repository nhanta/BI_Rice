# Guideline for Running Code

#

**1 Data**

_Xcont.csv_: GWAS data of Oryza Sativa

_cont\_LMM\_grain.weight\_p0.05\_sort.tsv_: grain weight data after filtering by biological characteristics,

_cont\_LMM\_time.flowering\_p0.05\_sort.tsv_: time to flowering data after filtering by biological characteristics,

_Ycont.txt_: output data including phenotypes of time to flowering and grain weight,

_indep\_1000\_10\_0.3.prune-43k.in_: input data for advanced regressor after filtering by biological characteristics.

We have compressed them into a file _Prepared\_data.rar._

**2 Filtering Data**

Python code: **BI\_Rice\_SNPs\_Pvl\_Filtering.py**

_The code is run with Python 3.7.1. Pandas package need to be installed before running the code._

Using the data from _Prepared\_data.rar_.

Filtering the data with p value, then put it into the models for training. We have also saved the results in _Export\_data.rar._

**3 Training Model**

Python code: **BI\_Rice\_SNPs.py**

_The code is run with Python 3.7.1. Pandas package and scikit-learn package need to be installed before running the code._

The data from _Export\_data.rar_ is used to train the model.


3.1 Grid Search with cross-validation: [sklearn.model\_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

Link: https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.GridSearchCV.html

Used with random forest regression and support vector regression.

3.2 Radom forest regression: [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

Link: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

3.3 Support vector regression: [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

Link: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

3.4 Lasso with cross-validation: [sklearn.linear\_model.LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)

Link: https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.LassoCV.html

3.5 Multi-task Lasso with cross-validation: [sklearn.linear\_model.MultiTaskLassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html)

Link: https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.MultiTaskLassoCV.html

3.6 Elastic Net with cross-validation: [sklearn.linear\_model.ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)

Link: https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.ElasticNetCV.html

3.7 Multi-task Elastic Net with cross-validation: [sklearn.linear\_model.MultiTaskElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html)

Link: https://scikit-learn.org/stable/modules/generated/sklearn.linear\_model.MultiTaskElasticNetCV.html

The results have been compressed into the file _Results.rar._

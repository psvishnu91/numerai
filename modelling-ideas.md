### Intro

Here we will build the first version of the model. The steps involved here are
1. load the data with subsampling.
2. take the medium featureset.
3. build xgboost model and see feature importance with shap.
4. blow up the feature space and feature select.
   1. Take top n features, do some transforms?
   2. piecewise linear
5. build another xgb model.

### Next steps
#### Immediate
Build a big model we can submit.
1. [Not started] Setup a distributed training infra with dask 
2. [Not started] Find 50 riskiest features according to numerai logic and save to file.
3. [Not started] Train model using numerai hyperparams and submit model.
4. [Not started] Hyperparam search over model params one at a time.

#### Longer term
1. [Not started] Using best global hyperparams, find feature importance by era using shapley values.
   Find features that have the highest variations in feature importance rank and neutralise
   them instead.
2. [Not started] Create model on all eras. Delete worst performing eras in the first half of the data.
3. [Not started] Instead of equal weights on targets regress on target or sharpe value
4. [Not started] Can we construct feature suppression as a penalty term in the loss?

## Ideas / Open Questions

### 1. Sharpe loss
WKT that diversification reduces portfolio risk (variance) even when choosing amongst assets with
the same expected return and same variances (as long as they are not perfectly correlated).
Could we incorporate this risk computation in the loss function optimised by GBT? Can we use
sharpe ratio as the GBT loss function? `mdo` already has a post on this.

### 2. Neutralisation linearity
* Why should the neutralisation be linear?
* There must be other ways to neutralise.

### 3. Learning curve with increasing epochs of training and performance


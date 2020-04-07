from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# import preprocessing functions
import preprocessors as pp

# import list variables
import config


price_pipe = Pipeline(
    [
        ('temporal_variable',
            pp.TemporalVariableEstimator(
                variables=config.TEMPORAL_VARS,
                reference_variable=config.DROP_FEATURES)),

        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),

        ('log_transformer',
            pp.LogTransformer(variables=config.NUMERICALS_LOG_VARS)),

        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),

        ('scaler', MinMaxScaler()),
        ('lasso', Lasso(alpha=0.005, random_state=0))
    ]
)

import numpy as np
from sklearn import linear_model

class MakeSeasonTrendModel(object):

    """Class containing all information and functions relating to fitting a 
    season-trend model to a single pixel."""
    
    def __init__(self, datetimes):
        
        self.T = 365.25
        self.pi_val_simple = (2 * np.pi) / self.T
        self.pi_val_advanced = (4 * np.pi) / self.T
        self.pi_val_full = (6 * np.pi) / self.T
        self.datetimes = datetimes
        
        self.lasso_model = None
        self.residuals = None
        self.RMSE = None
        self.coefficients = None
        self.predicted = None
        self.alpha = None     # Needed to store alpha if CV is used
        self.num_obs = len(datetimes)

    def fit_model(self, band_data, cv=True, alpha=1):
        
        """Given a 1D time series of values, fit a Lasso model to the data and 
        store the resulting model coefficients and Root Mean Square Error."""
        
        # Rescale dates to start from 0
        rescaled = self.datetimes - self.datetimes[0]
        
        # Complexity of fit is based on Zhu et al. 2015: Generating synthetic Landsat images based on all available Landsat data: Predicting Landsat surface reflectance at any given time.
        # There should be at least three times more data points that the number of coefficients.
        
        # Less than 18 observations but at least 12. Fit one harmonic term (simple model, four coefficients inc. intercept)
        x = np.array([rescaled,
                      np.cos(self.pi_val_simple * rescaled),
                      np.sin(self.pi_val_simple * rescaled)])
        
        # 18 or more observations. Fit two harmonic terms (advanced model, six coefficients)
        if(self.num_obs >= 18):
            x = np.vstack((x, np.array([np.cos(self.pi_val_advanced * rescaled),
                      np.sin(self.pi_val_advanced * rescaled)])))
        
        # 24 or more observations. Fit three harmonic terms (full model, eight coefficients)
        if(self.num_obs >= 24):
            x = np.vstack((x, np.array([np.cos(self.pi_val_full * rescaled),
                      np.sin(self.pi_val_full * rescaled)])))
    
        x = x.T

        if(cv): # If cross validation should be used to find alpha parameter
            self.lasso_model = linear_model.LassoCV(fit_intercept=True).fit(x, band_data)
            self.alpha = self.lasso_model.alpha_
        else:
            self.lasso_model = linear_model.Lasso(fit_intercept=True, alpha=alpha).fit(x, band_data)
            self.alpha = alpha
                                
        self.predicted = self.lasso_model.predict(x)
        
        self.coefficients = self.lasso_model.coef_
        
        self.residuals = band_data - self.predicted
    
        # Get overall RMSE of model
        self.RMSE = np.sqrt(np.mean(self.residuals ** 2))
    









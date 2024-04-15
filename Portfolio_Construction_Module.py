import numpy as np
import pandas as pd
from scipy.optimize import minimize


def sample_cov(r, t, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()*t

def sample_cov_lw(r, t, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf().fit(r.dropna())
    cov_df = pd.DataFrame(cov.covariance_, index = r.columns, columns = r.columns)
    return cov_df*t

def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal,eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec

def calculate_corr_m(variance_df, columns = None):
    # 
    number_of_assets = len(variance_df.columns)
    Rho = np.ones((number_of_assets,number_of_assets))  # Matriz de correlaciones
    for i in range(number_of_assets):  
        j = i+1
        while j < number_of_assets:
            Rho[i,j] = variance_df.iloc[i,j]/np.sqrt(variance_df.iloc[i,i]*variance_df.iloc[j,j])
            Rho[j,i] = Rho[i,j]
            j+=1
    if(columns is not None):
        Rho = pd.DataFrame(Rho, index=columns, columns=columns)
    return Rho

def mpPDF(var,q,pts):
    # Marcenko-Pastur pdf
    # q=T/N
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf,index=eVal)
    return pdf

from sklearn.neighbors import KernelDensity

def fitKDE(obs,bWidth=.25,kernel="gaussian",x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None:x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:x=x.reshape(-1,1)
    logProb=kde.score_samples(x) # log(density)
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

def errPDFs(var,eVal,q,bWidth,pts=1000):
    # Fit error
    pdf0=mpPDF(var,q,pts) # theoretical pdf
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal,q,bWidth):
    # Find max random eVal by fitting Marcenko’s dist
    out=minimize(lambda *x:errPDFs(*x),.5,args=(eVal,q,bWidth),
    bounds=((1E-5,1-1E-5),))
    if(out["success"]):
        var=out["x"][0]
    else:
        var=1
    eMax=var*(1+(1./q)**.5)**2
    return eMax,var

def denoisedCorr(eVal,eVec,nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T)
    corr1=calculate_corr_m(corr1)
    return corr1


def deNoiseCov(cov0,q,bWidth):
    corr0=calculate_corr_m(cov0)
    eVal0,eVec0=getPCA(corr0)
    eMax0,var0=findMaxEval(np.diag(eVal0),q,bWidth)
    nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1=denoisedCorr(eVal0,eVec0,nFacts0)
    cov1=calculate_corr_m(corr1,np.diag(cov0)**.5)
    return cov1

def ewma_covariance(r,t, alpha=0.94, **kwargs):
    """
    Estimate the covariance matrix using Exponentially Weighted Moving Average (EWMA).
    
    Parameters:
    r (pandas.DataFrame): Input data, where rows are observations and columns are variables.
    alpha (float): Smoothing parameter (0 < alpha <= 1), default is 0.94.
    
    Returns:
    pandas.DataFrame: Estimated covariance matrix.
    """
    mean = r.mean(axis=0)
    demeaned_data = r - mean
    cov_matrix = demeaned_data.T @ demeaned_data / len(r)
    
    # Initialize the EWMA covariance with the simple covariance
    ewma_cov = cov_matrix
    
    for i in range(1, len(r)):
        # Update the EWMA covariance using the previous estimate and the current observation
        ewma_cov = alpha * np.outer(demeaned_data.iloc[i], demeaned_data.iloc[i]) + (1 - alpha) * ewma_cov
    
    return pd.DataFrame(ewma_cov, index=r.columns, columns=r.columns)*t


def sample_ret(r, t, **kwargs):
    """
    Returns the sample mean of the supplied returns
    """
    return r.mean()*t

def sample_ret_trailing(r, t, **kwargs):
    """
    Returns the sample mean of the supplied returns
    """
    return r.iloc[(t-1):,:].mean()*t

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


### Include the GMV 
### 

import gurobipy as gp
from gurobipy import GRB

# There is one Little Modification. 

def gmv(cov, ub_list = None, lb_list = None):

    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    assets_names = cov.index.to_list()
    
    if ub_list == None:
        ub_list = np.ones(n)
    if lb_list == None:
        lb_list = np.zeros(n)

    m = gp.Model()
    w = m.addMVar(n, lb = lb_list, ub = ub_list, name = assets_names )

    # Min Variance
    ones = np.ones(n)
    cov = cov.to_numpy()
    
    m.addConstr(ones @ w == 1)
    m.setObjective(w @ cov @ w)
    m.optimize()
    
    results = []
    for i in range(len(assets_names)): 
        results.append(m.getVars()[i].X)
    optimal_values = pd.Series(results, index = assets_names)
    optimal_values

    # The returns are on the base of a 
    return optimal_values

def TargetVol(returns, cov, target_vol, ub_list = None, lb_list = None, silent = False):
    """
    Returns the weight of the Markowitz Portfolio Optimization
    given a set of assets returns and a covariance matrix 
    and a target volatility
    """
    # This line is used in order to check the inputs that the optimization is receiving. 
    #cov.join(returns.rename('returns')).rename_axis(str(target_vol)).to_csv('parameters2.csv')
    n = cov.shape[0]
    

    if(ub_list == None):
        ub_list = np.ones(n)
    else:
        ub_list = np.array(ub_list) 
    if(lb_list == None):
        lb_list = np.zeros(n)
    else: 
        lb_list = np.array(lb_list)
    
    ub_lb = pd.DataFrame([ub_list, lb_list], index = ['ub_list', 'lb_list'], columns = cov.index.to_list())
    ub_lb.to_csv('lb_ub.csv')
    assets_names = cov.index.to_list()
    
    
    m = gp.Model()
    w = m.addMVar(n, lb = lb_list, ub = ub_list, name = assets_names )

    # 
    ones = np.ones(n)
    cov = cov.to_numpy()
    returns = returns.to_numpy()
    
    m.addConstr(ones @ w == 1)
    m.addConstr(w @ cov @ w <= target_vol**2)
    m.setObjective(returns @ w, GRB.MAXIMIZE)
    m.optimize()
    
    results = []
    for i in range(len(assets_names)): 
        results.append(m.getVars()[i].X)
    optimal_values = pd.Series(results, index = assets_names)
    optimal_values
    
    return optimal_values


def weight_gmv(r, cov_estimator=sample_cov, t = 1, lb_list = None , ub_list = None , **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, t, **kwargs)
    return gmv(est_cov)

def weight_target_volatility(r, cov_estimator=sample_cov, ret_estimator = sample_ret, t = 1, target_vol = 0.1, lb_list = None , ub_list = None , **kwargs):
    """
    Produces the weights of the Efficient Frontier given a covariance matrix of the returns
    """
    est_cov = cov_estimator(r, t, **kwargs)
    est_ret = ret_estimator(r, t, **kwargs)
    
    return TargetVol(est_ret, est_cov, target_vol, ub_list, lb_list)

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

def risk_contribution(w,cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio weights and a covariance matrix
    """
    total_portfolio_var = portfolio_vol(w,cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
    return risk_contrib

def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        """
        Returns the Mean Squared Difference in risk contributions
        between weights and target_risk
        """
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)


def weight_erc(r, cov_estimator=sample_cov, t = 1, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, t, **kwargs)
    return equal_risk_contributions(est_cov)

def backtest_ws(r, estimation_window=60, weighting=weight_ew, verbose=False, **kwargs):
    
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    
    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    weights = []
    for win in windows:
        pd.DataFrame(win, index = [1, 2], columns = [1]).to_csv('win.csv')
        weights.append(weighting(r.iloc[win[0]:win[1]], **kwargs)) 
    # convert List of weights to DataFrame
    
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window:].index, columns=r.columns)    
    arithmetic_returns = r.apply(np.exp)-1
    returns = (weights * arithmetic_returns).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    
    return weights, returns

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3



def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x



def minimize_vol(target_return, er, cov, lb_list, ub_list, assets_names):
    
    expected_returns = target_return


    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model()
    number_of_assets = len(cov)
    w = m.addMVar(len(cov), lb = lb_list, ub = ub_list, name = assets_names )
    ones = np.ones(number_of_assets)

    m.addConstr(ones @ w == 1)
    # New noncanonical constraint
    m.setObjective(w @ cov @ w)
    m.optimize()

    m.addConstr(er @ w >= rmin)
    m.optimize() 
    return 
        

def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """


    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    rhos = r.corr()
    n = rhos.shape[0]
    # this is a symmetric matrix with diagonals all 1 - so the mean correlation is ...
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    return pd.DataFrame(ccor * np.outer(sd, sd), index=r.columns, columns=r.columns)



def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample

def compute_rolling_returns(returns_series, window=12):
    """
    Computes the rolling 1-year returns of a monthly return series.

    :param returns_series: A Pandas Series containing monthly returns.
    :param window: The rolling window size in months (default is 12 for 1 year).
    :return: A Pandas Series containing the rolling 1-year returns.
    """
    # Convert monthly returns to cumulative returns
    cumulative_returns = (1 + returns_series).cumprod()

    # Compute rolling 1-year returns
    rolling_returns = cumulative_returns.pct_change(periods=window - 1)

    return rolling_returns


def estimate_outperformance_probability(indexes_returns, benchmark_returns, horizon):
    """
    Estimates the probability that each index in a DataFrame outperforms a benchmark over different horizons.

    :param indexes_returns: A Pandas DataFrame containing returns of multiple indexes.
    :param benchmark_returns: A Pandas Series containing returns of the benchmark.
    :param horizon: The comparison horizon ('1Y' for 1 year, '3Y' for 3 years, etc.).
    :return: A Pandas Series containing the probability of outperformance for each index.
    """
    # Convert the horizon to a rolling window size in months
    horizon_map = {'6M': 6, '1Y': 12, '3Y': 36}
    if horizon not in horizon_map:
        raise ValueError("Unsupported horizon. Please use '1Y' or '3Y'.")
    window = horizon_map[horizon]

    # Compute rolling returns for the indexes and the benchmark
    rolling_indexes_returns = indexes_returns.apply(lambda x: compute_rolling_returns(x, window))
    rolling_benchmark_returns = compute_rolling_returns(benchmark_returns, window)

    # Calculate the differences in rolling returns between each index and the benchmark
    return_differences = rolling_indexes_returns.subtract(rolling_benchmark_returns, axis=0)

    # Count the number of periods where each index outperforms the benchmark
    outperformance_count = (return_differences > 0).sum()

    # Calculate the probability of outperformance for each index
    probability = outperformance_count / len(return_differences.dropna())

    return probability


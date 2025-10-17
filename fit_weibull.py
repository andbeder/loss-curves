import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Weibull cumulative distribution function - explicitly calculated
# Formula: f(t, k, lambda) = 1 - e^(-(t/lambda)^k)
def weibull_cdf(t, k, l):
    """
    Explicitly calculate Weibull CDF without using built-in functions.
    t: time points (array or scalar)
    k: shape parameter (must be > 0)
    l: scale parameter (lambda, must be > 0)
    """
    # Handle t=0 case explicitly
    if np.isscalar(t):
        if t == 0:
            return 0.0
        else:
            return 1.0 - np.exp(-((t / l) ** k))
    else:
        # For arrays
        result = np.zeros_like(t, dtype=float)
        non_zero = t > 0
        result[non_zero] = 1.0 - np.exp(-((t[non_zero] / l) ** k))
        return result

def fit_weibull(row, t_points):
    """
    Fit Weibull parameters k and lambda to minimize mean squared error.
    Uses multiple optimization strategies to match Excel Solver results.
    """
    y = row.values.astype(float)

    # Loss function: mean squared error (same as Excel Solver minimizes)
    def loss(params):
        k, l = params
        if k <= 0 or l <= 0:
            return 1e10  # Large penalty for invalid parameters
        try:
            y_pred = weibull_cdf(t_points, k, l)
            mse = np.mean((y - y_pred) ** 2)
            return mse
        except (OverflowError, RuntimeWarning):
            return 1e10

    # Try multiple optimization approaches to find global minimum
    # (Excel Solver uses GRG Nonlinear which can find different local minima)

    best_result = None
    best_loss = np.inf

    # Method 1: Multiple starting points with L-BFGS-B (similar to Excel's GRG Nonlinear)
    initial_guesses = [
        [1.5, 3.0],   # Default starting point
        [2.0, 4.0],   # Alternative starting points
        [0.5, 2.0],
        [3.0, 5.0],
        [1.0, 1.0],
    ]

    for x0 in initial_guesses:
        try:
            res = minimize(loss, x0, method='L-BFGS-B',
                          bounds=[(0.01, 10), (0.01, 100)])
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_result = res.x
        except:
            continue

    # Method 2: Try with SLSQP (another gradient-based method)
    try:
        res = minimize(loss, [1.5, 3.0], method='SLSQP',
                      bounds=[(0.01, 10), (0.01, 100)])
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_result = res.x
    except:
        pass

    # Method 3: Differential Evolution for global optimization (more thorough)
    # This helps find better global minima when local methods get stuck
    try:
        res = differential_evolution(loss, bounds=[(0.01, 10), (0.01, 100)],
                                     seed=42, maxiter=300, atol=1e-7, tol=1e-7)
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_result = res.x
    except:
        pass

    if best_result is not None:
        return best_result
    else:
        return (np.nan, np.nan)

def main():
    # Read Excel file
    df = pd.read_excel('Fake Loss Curves.xlsx')
    # Adjust these column names/indexes as needed
    data_cols = df.columns[1:8]  # B through H (years 1-7)
    t_points = np.arange(1, len(data_cols) + 1)  # [1, 2, 3, 4, 5, 6, 7]

    print(f"Fitting Weibull parameters for {len(df)} rows...")
    print(f"Data columns (time points): {list(data_cols)}")
    print(f"Time values: {t_points}")
    print()

    k_list = []
    l_list = []

    for idx, row in df[data_cols].iterrows():
        if idx % 10 == 0:
            print(f"Processing row {idx}/{len(df)}...")
        k, l = fit_weibull(row, t_points)
        k_list.append(k)
        l_list.append(l)

    df['k'] = k_list
    df['lambda'] = l_list

    print(f"\nFitting complete!")
    print(f"Sample results (first 5 rows):")
    print(df[['k', 'lambda']].head())
    print(f"\nWriting results to weibull_fitted_output.xlsx...")

    # Write to Excel to preserve data types (especially text columns)
    df.to_excel('weibull_fitted_output.xlsx', index=False, engine='openpyxl')
    print("Done!")

if __name__ == '__main__':
    main()

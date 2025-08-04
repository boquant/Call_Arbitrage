# Local Volatility Surface Fitting & Pricing using SVI and Crank–Nicolson

## 1. Data Preprocessing

**Function:** 
```python
option_info(symbol, months, moneyness, liquidity, r, q)
```

Select the most liquid, near-expiry, near-ATM option chains for SVI fitting and robustness testing.

### Parameters:
- `symbol`: Stock symbol (e.g., `'AAPL'`)
- `months`: Collect options with expiration ∈ (1, months), skipping first-month expiry
- `moneyness`: Range of strikes ∈ \[F · (1 - m), F · (1 + m)], where F is forward price
- `liquidity`: Take top x% liquid options in the above range
- `r`: Risk-free rate
- `q`: Dividend rate

### Outputs:
- `iv_dict[T] = {K: IV}`
- `price_dict[T] = {K: price}`
- `filtered_dict[T] = np.array([[K₁, K₂, ...], [IV₁, IV₂, ...]])`
- `spot_price`: current market spot

---

## 2. Estimate SVI Parameters

### SVI Parameterization

The Stochastic Volatility Inspired (SVI) model expresses total implied variance $w(k) = \sigma_{\text{BS}}^2 \cdot T$ as a function of log-moneyness $k = \log(K / F)$ , where $F = S \cdot e^{(r - q)T}$ is the forward price.

The SVI formula is:
```math
w(k) = a + b \left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2} \right]
```

- k : log-moneyness  
-  a : minimum total variance  
-  b : slope (controls angle of wings)  
- $\rho$ [-1, 1] : correlation-like parameter, skews the smile  
- m : horizontal shift (center of smile)  
-  $\sigma$ : curvature (controls the width of the smile)  

The function \( w(k) \) represents the total implied variance for a given log-moneyness \( k \), and is fit independently for each maturity \( T \).


### Objective Function

To calibrate the model, we minimize the squared difference between market-observed implied variances \( \sigma^2_{\text{market}} \cdot T \) and the SVI variance:
```math
\min_{\theta = (a, b, \rho, m, \sigma)} \sum_{i=1}^{N} \left( w(k_i; \theta) - \sigma_i^2 T_i \right)^2

```
This is implemented in:

- `svi_objective(params, S, K, T, r, q, iv)`: returns the loss function value for a given set of SVI parameters
- `svi_lv_fit(S, K, T, r, q, iv)`: minimizes the loss using numerical optimization (e.g., L-BFGS-B)

Each maturity is fit independently to produce a calibrated volatility smile/skew.

---
### 3. Local Volatility Estimation

**Functions:**
```python
- `svi_w(k, params)`, `svi_w_k(k, params)`, `svi_w_kk(k, params)`
- `local_volatility(K, T, s, r, q, svi_params_dict)`
```

### Gatheral's Dupire Equation (SVI-based)

Use the Dupire equation (improved by Gatheral) to derive local variance from implied variance surface:
```math

\sigma^2_{\text{loc}}(K, T) = 
\frac{ \frac{\partial w}{\partial T} }
{ \left( 1 - \frac{k}{w} \frac{\partial w}{\partial k} \right)^2 -
 \frac{1}{2} \left( \frac{\partial^2 w}{\partial k^2} - \frac{1}{w} 
\left( \frac{\partial w}{\partial k} \right)^2 \right) }
```

**Where:**
```math
 w(k, T) = \sigma^2_{\text{BS}}(k, T) \cdot T
```
```math
 k = \log(K / F) 
```
Interpolate across different maturities  T  using linear interpolation to estimate volatility on arbitrary expirations.

>  **Note**: We **do not take the square root** here, since local volatility is used in squared form in pricing.

---

### 4. Crank–Nicolson Finite Difference Pricing

**Function:**

```python
Crank_Nicolson_Pricing(spot_price, T, K, r, q, sigma_sq, n_x=200, n_t=200, n_std=3)
```


### Mathematical Idea:

Transform Black–Scholes PDE using change of variable $x = \log(S)$:

```math
\frac{\partial V}{\partial t} = \frac{1}{2} \sigma^2_{\text{loc}}(x, t) \frac{\partial^2 V}{\partial x^2}
+ \left( r - q - \frac{1}{2} \sigma^2_{\text{loc}}(x, t) \right) \frac{\partial V}{\partial x} - rV
```

Apply Crank–Nicolson discretization in time and space to solve backward from payoff.


### Parameters:

- `sigma_sq`: Squared local volatility matrix.
- `n_std`: Domain size = $[\mu - n * std, \mu + n * std]$, where $\mu = \log(S)$
- `n_x`, `n_t`: Grid resolution in space and time.

### Discretization:

Let $x_i = x_{\min} + i \cdot \Delta x$, $t_n = n \cdot \Delta t$. 
Then CN (Crank–Nicolson) scheme is the average of explicit and implicit Euler:
```math

A \cdot V^{n+1} = B \cdot V^n
```

Where A and B are tridiagonal matrices:
```math

A = I - \frac{\Delta t}{2} \cdot \mathcal{L}
```
```math
B = I + \frac{\Delta t}{2} \cdot \mathcal{L}
```

$\mathcal{L}$ is the differential operator:

```math
\mathcal{L} V = a_i V_{i-1} + b_i V_i + c_i V_{i+1}

```
With:
```math

a_i = \frac{1}{2} \left[ \left( \frac{\sigma_i^2}{\Delta x^2} \right) - \left( \frac{\mu_i}{\Delta x} \right) \right]
```
```math

b_i = - \left[ \frac{\sigma_i^2}{\Delta x^2} + r \right]
```
```math
c_i = \frac{1}{2} \left[ \left( \frac{\sigma_i^2}{\Delta x^2} \right) + \left( \frac{\mu_i}{\Delta x} \right) \right]

```
Where:
```math
 \mu_i = r - q - \frac{1}{2} \sigma_i^2
```
```math
 \sigma_i^2 = \sigma^2_{\text{loc}}(x_i, t_n) 
```

Solve the tridiagonal system at each time step using **Thomas algorithm**.

---

### 5. Robustness Testing

**Function:**

```python
test_robustness(symbol, test_T_index, months=6, moneyness=0.2, liquidity=0.5,
                rfr=0.02, dividend=0, indices=0, n_x=5, n_t=5, n_std=3)
```

Test fitted local volatility model using out-of-sample prices:

- **Compare:**
  - Market price and implied vol
  - Fitted Crank–Nicolson price
  - Interpolated price from nearby strikes

- **Optional:** Exclude deep ITM/OTM options with `indices` parameter

**Visualization:** Plot test comparisons for price and IV surface.

<img width="788" height="990" alt="image" src="https://github.com/user-attachments/assets/34e89e04-2430-49d8-b2a2-6b43e19ce502" />

---

### Known Issues

1. **Instability at Extreme Strikes**: Dupire’s denominator may diverge. Mitigated partially through interpolation and smoothing.
2. **CN Underestimation Near ATM**: Low local vol & narrow domain lead to artificially small option values.  
   Recommend applying domain extension or regularization.

---

### Future Improvements

- Allow spatial grid in original \( S \) space, not \( \log(S) \).
- Test stochastic volatility models (e.g., Heston, SABR) as alternative.
- Improve Dupire denominator handling via spline smoothing or neural nets.
- Fit full surface across both strike and expiry using global SVI parameterization.

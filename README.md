# Numerai learning

This repo contains materials for getting started on Numerai's ML problems. Topics include
1. Convex optimisation
2. ML modelling of numerai tournament

## Resources
### Convex optimisation for portfolio allocation
1. [Ipython notebook](https://github.com/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb) from [Boyd's Stanford short course](https://web.stanford.edu/~boyd/papers/cvx_short_course.html)
2. [Slides](https://web.stanford.edu/~boyd/papers/pdf/cvx_applications.pdf) on the application
  of convex optimisation.
3. Linear algebra [Quadratic form](https://sites.millersville.edu/rumble/Math.422/quadform.pdf) 

## Convex optimisation

### Portfolio maximisation formulation
Read through
* the [ipython notebook](https://github.com/cvxgrp/cvx_short_course/blob/master/applications/portfolio_optimization.ipynb) from [Boyd's Stanford short course](https://web.stanford.edu/~boyd/papers/cvx_short_course.html)
* [slides](https://web.stanford.edu/~boyd/papers/pdf/cvx_applications.pdf) on the application
  of convex optimisation.

The following is almost an exact rewrite of the notebook.

We formulate the portfolio allocation across a set of assets as a convex optimisation problem where we want to maximise return and minimise risk.

We have a fixed budget normalised to $\mathbf{1}$ and $\mathbf{1}^T w = 1$ where $w$ is the allocation to each stock. If $w \in R_+$ then we have long only positions.

$$\text{leverage} = ||w||_1 = \mathbf{1}^Tw_+ + \mathbf{1}^T w_-$$

We can short sell and use that money to go long. So we can technically invest more than our total budget.

#### Asset return
Now we define asset return as $\frac{p_i^+-p_i}{p_i}$ where $p_i$ and $p_i^+$ is the stock price at the beginning and the end of the day respectively. For each of the $n$ stocks, we can compute this fraction across multiple days/months/years. Consequently, we can compute the mean asset return $\mathbb{E[r_i]}=mu_i$ and the Covariance matrix $\Sigma=\mathbb{E}\left[\left(\mathbf{r}-\mu\right)\left(\mathbf{r}-\mu\right)^T\right]$. The reason we care for the covariance is that two stocks can vary together (we ignore higher order moments). Recall that the covariance matrix is positive semi-definite, ie., $\mathbf{z}^T \Sigma \mathbf{z} \geq 0, \; \; \forall \mathbf{z} \in \mathbf{R}^n$, that is $\Sigma$ rotates any $z$ by an acute angle.

#### Portfolio Return
We define expected portfolio return $\mathbb{E}[R]=\mu^T w$ and portfolio variance as $\mathbf{Var}(R) = w^T \Sigma w = \sum_i \sum_j (w_i^2 \mathbf{var}(r_i) + w_j^2 \mathbf{var}(r_j) + w_i w_j \mathbf{cov}(r_i, r_j))$, where $i, j \in [1, n]$ are the indices of $\Sigma$. **Risk** is defined as $\mathbf{std dev}(R) = \sqrt{\mathbf{Var}(R)}$.

#### Objective
We want to maximise return while minimising risk. We can set this up as the **classical Markowitz portfolio optimisation**.

$$\begin{align*}
    \arg\max_{\mathbf{w}} \mathbf{w^T\mu} &- \gamma \mathbf{w^T} \Sigma \mathbf{w}   \\
    \text{such that,    } \mathbf{1^Tw}=1 &, w \in \mathcal{W}
\end{align*}$$

#### Risk vs Return trade-off curve
Here $\gamma > 0$ is the **risk aversion factor**. As we increase $\gamma$ and our risk aversion, our expected return will diminish along with risk. We can trace that curve and choose what we determine as the optimal *risk-return trade-off*.

![](assets/images/readme/cvx_risk_retn_tradeoff.svg)

#### Choosing the optimal risk-return trade-off
Above, we approximate the portfolio return by a normal distribution. We can now plot the return at different risk-aversion values $\gamma$ and choose the one we are most comfortable with.
![](assets/images/readme/cvx_risk_retn_dist.svg)

#### Other constraints
1. Max leverage constraint $|w|_1 < L^{max}$. As we increase leverage, we can
   get higher returns for the same risk (defined by stddev). Also higher leverage allows
   us to take higher risks. ![](assets/images/readme/cvx_risk_retn_leverage_max.svg)
2. Set max leverage and max risk as constraints and optimise return. Here we show
   optimal asset allocations for different maximum leverage for a fixed max risk.
   ![](assets/images/readme/cvx_allocn_fixed_risk.svg)
3. Market neutral constraint. We want the portfolio returns to be uncorrelated with the
    market returns. $m^T \Sigma w = 0$  where $m_i$ is the capitalisation of the asset
    $i$ and $M=\mathbf{m}^T\mathbf{r}$ is the market return. That is $m_i$ is the
    fraction of the market cap this asset holds. Think what fraction of S&P500 does
    Apple hold. $m^T \Sigma w = \mathbf{cov(M, R)}$  by setting this to zero, we make
    the portfolio uncorrelated with the market. The allocation $w$ vector is rotated
    orthogonal to the market cap fraction vector
    $m \perp  \Sigma w \implies m^T \cdot (\Sigma w) = 0$.

#### Variations
1. Fix minimum return $R^{min} \geq \mathbf{\mu}^T w$ and minimise risk ($w^T \Sigma w$).
2. Include broker costs for short positions $\mathbf{s}^T w_-$.
3. Include transaction fee to change from current portfolio as a penalty
   $\kappa^T |w - w^{cur}|^{\eta}, \kappa \geq 0$. Usual values for $\eta$ are $\eta = 0, 1.5, 2$.
4. Factor covariance model explained later.

#### Variation - Factor Covariance model
In a factor covariance model, we assume
1. Each of the $n$ stocks belong to $k$ factors ($k	\ll n, k \approx 10$) with different proportions (linear weighting/affine). A factor can be thought of as an industry sector (tech vs energy vs finance etc.)
2. Individual stocks are not directly correlated with other stocks but only indirectly
   through their factors.

We can thus factorise the Covariance matrix $\Sigma$ as $\Sigma =  F \tilde\Sigma F^T + D =  F_{[n\times k]} \tilde\Sigma_{[k \times k]} F^T_{[k \times n]} + D_{[n \times n]}$, where 
- $F_{[n\times k]}$ is the _factor loading matrix_ and $F_{ij}$ is the loading of asset $i$ to factor $j$ and
- $D_{[n \times n]}$ is a diagonal matrix where $D_{ii}>0$ with the individual risk of each stock independent of the factor covariance.
- $\tilde\Sigma_{[k \times k]} > 0$ is the factor covariance matrix (positive definite)

Portfolio factor exposure: $f = F^T w \in R^k$. This is a linear weighted sum of the factor
exposures of the fractional assets in the portfolio. To be factor neutral across all factors,
we need $(F^T_{[k\times n]} w_{[n\times 1]})_j=0, \; \forall j \in [1, k]$.

##### Formulation

$$
\begin{align*}
    \arg \max_{w} \mathbf{\mu^T} w &- \gamma \left[(F^Tw)^T \tilde \Sigma (F^Tw) + w^T D w\right] \\
    \implies \arg \max_{w} \mathbf{\mu^T} w &- \gamma \left[f^T \tilde \Sigma f + w^T D w\right] \\
    \text{such that, }f_{[k\times 1]} &= F^Tw \\
    \mathbf{1}^Tw=1,\;  &w\in \mathcal{W}, f \in \mathcal{F}
\end{align*}
$$

Computational complexity to solve the problem falls from $O(n^3)$ to $O(nk^2)$.

In order to leverage this we define the problem like below

<details>
    <summary>Python code for standard and factor portfolio optimisation</summary>

``` python
n = 3000  # number of stocks / assets
m = 50  # number of factors
np.random.seed(1)
mu = np.abs(np.random.randn(n, 1))  # average return = (p_new - p)/p over days
Sigma_tilde = np.random.randn(m, m)  # factor cov matrix Σ'
Sigma_tilde = Sigma_tilde.T.dot(Sigma_tilde)
D = sp.diags(np.random.uniform(0, 0.9, size=n))  # Stock idiosyncratic risk (indep of factor)
F = np.random.randn(n, m)  # Factor loading matrix: how much each stock relates to a factor

# Standard model portfolio optimisation
#######################################

w = cp.Variable(n)  # portfolio allocation
gamma = cp.Parameter(nonneg=True)  # Risk aversion parameter
Lmax = cp.Parameter(nonneg=True)  # Maximum leverage
Sigma = F @ Sigma_tilde @ F.T + D  # [n*n] = [n*k] x [k*k] x [k*n] + [n*n]
preturn = mu.T @ w  # [1*1] = [1*n] x [n*1]
risk = w.T @ Sigma @ w  # [1*1] = [1*n] x [n*n] x [n*1]
# (or equivalently) risk = cp.quad_form(w, Sigma)
prob_std = cp.Problem(
    objective=cp.Maximize(preturn - gamma * risk),
    constraints={cp.sum(w)==1, cp.norm(w, 1) <= Lmax},
)
# Solve it: Takes 15 minutes
gamma.value = 0.1
Lmax.value = 2
prob.solve(verbose=True)
print(f"Return: {preturn.value[0]}, Risk: {np.sqrt(risk.value)}")

# Factor model portfolio optimisation
#####################################
w = cp.Variable(n)
# Make the exposure a variable and constraint it to F^T w = f
f = cp.Variable(m)
gamma, Lmax = cp.Parameter(nonneg=True), cp.Parameter(nonneg=True)
retn_factor = mu.T @ w
risk_factor = cp.quad_form(f, Sigma_tilde) + cp.sum_squares(np.sqrt(D) @ w)
# (or equivalently) risk_factor = f.T @ Sigma_tilde @ f + w.T @ D @ w
prob_factor = cp.Problem(
    objective=cp.Maximize(retn_factor - gamma * risk_factor),
    constraints={
        cp.sum(w)==1,
        cp.norm(w, 1) <= Lmax,
        F.T@w==f,  # NOTE: This is a new constraint
    },
)
# Solve it: Takes 1.2 seconds
gamma.value = 0.1
Lmax.value = 2
prob.solve(verbose=True)
print(f"Return: {preturn.value[0]}, Risk: {np.sqrt(risk.value)}")
print(prob_factor.solver_stats.solve_time)
```
</details>

## Ideas / Open Questions

WKT that diversification reduces portfolio risk (variance) even when choosing amongst assets with
the same expected return and same variances (as long as they are not perfectly correlated).
Could we incorporate this risk computation in the loss function optimised by GBT? Can we use
sharpe ratio as the GBT loss function?
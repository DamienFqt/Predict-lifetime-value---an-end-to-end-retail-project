# Data Generating Process

The data is generated as follows.

We assume $n$ customers at the beginning of the period.

### Assumptions

- **(H1)** Customers can leave the company, but no new customers can enter.
- **(H2)** Customers are independent and identically distributed, conditional on their individual heterogeneity.

---

## Customer Heterogeneity

Customers are heterogeneous. Each customer $i \in \{1,\dots,n\}$ is assigned a latent heterogeneity parameter:

$$
\lambda_i \sim \Gamma(a,b),
$$

where $a,b \in \mathbb{R}$ and $\Gamma$ denotes the Gamma distribution.

---

## Yearly Data Generation

For each simulation year and each customer $i \in \{1,\dots,n\}$, the following variables are generated.

### Number of purchases

The number of purchases is conditional on customer heterogeneity:

$$
N_i \mid \lambda_i = \lambda \sim \mathcal{P}(\lambda),
$$

where $\mathcal{P}$ denotes the Poisson distribution.

---

### Raw satisfaction

Customer satisfaction depends on the number of purchases and can be equally positive or negative. Raw satisfaction is defined as the sum of $N_i$ independent uniform variables:

$$
S^{(\text{raw})}_i \mid N_i = n_i \sim n_i \times \mathcal{U}([0,5]).
$$

where $\mathcal$ denotes the (continuous) uniform distribution.

---

### Yearly satisfaction

Yearly satisfaction is capped at 5 (and is greater or equal than 0):

$$
S_i = \min\left(S^{(\text{raw})}_i, 5\right).
$$

---

### Average purchase amount

The average purchase amount depends positively on satisfaction. Conditional on satisfaction, it follows a log-normal distribution:

$$
X_i \mid S_i = s_i \sim \mathcal{LN}(c + d \cdot s_i, e),
$$

where $c,d\in \mathbb{R}$, $e>0$, $\mathcal{LN}$ denotes the log-normal distribution.

---

### Total yearly spending

Total yearly spending is given by:

$$
\text{tot}_i = N_i \cdot X_i.
$$

---

### Purchase dates

Conditional on the number of purchases, purchase dates are drawn uniformly over the year:

$$
T_{ij} \mid N_i \sim \mathcal{U}\left(\{1,\dots,n_{\text{days}}\}\right),
\quad j \in \{1,\dots,N_i\},
$$

where $n_{\text{days}} \in \{365,366\}$ depending on whether the year is leap or not.

- **(H3)** Purchases are uniformly distributed over the year (no seasonality, holiday, or weekend effects).

---

### Churn

A customer may churn at the end of the year, with a probability that increases when satisfaction is low:

$$
\text{churn}_i \sim \mathcal{B}\left(\frac{1}{1 + 8 S_i}\right),
$$

where $\mathcal{B}$ denotes a Bernoulli distribution.

---

## Time Indexing

The data is generated independently for each year. If a customer churns in a given year, all their future observations are removed.

We introduce a time index $t$ for all variables except heterogeneity:

$$
N_{it},\;
S^{(\text{raw})}_{it},\;
S_{it},\;
X_{it},\;
\text{tot}_{it},\;
T_{ijt},\;
\text{churn}_{it}.
$$

If $\text{churn}_{it} = 1$, all observations for customer $i$ at time $t+1$ and beyond are deleted.

- **(H4)** All years are independent.

---

## Recency

Recency is a key variable for modeling customer lifetime value.

Let:
- $T_{i1t}$ be the first purchase date of customer $i$ in year $t$,
- $T_{iN_{it}t}$ be the last purchase date of customer $i$ in year $t$.

Recency is defined as:

$$
R_{it} =
\begin{cases}
T_{i1(t+1)} - T_{iN_{it}t}, & \text{if } N_{i(t+1)} > 0, \\
\text{01/01}_{t+2} - T_{iN_{it}t}, & \text{otherwise}.
\end{cases}
$$

In words, recency measures the number of days between the last purchase of the current year and the first purchase of the following year. If the customer churns or makes no purchase the following year, recency is computed using January 1st of year $t+2$ in order to heavily penalize inactivity.

### Consequence

Recency is underestimated for customers who skip one or more years before returning.

For example, if a customer purchases on December 15th, 2021 and makes their next purchase on April 1st, 2023, their recency will be computed as $365 + 15$ instead of approximately $365 + 15 + 90$.

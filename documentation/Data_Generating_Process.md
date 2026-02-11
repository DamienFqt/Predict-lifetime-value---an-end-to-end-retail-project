# Data Generating Process

The data generating process is designed to mimic typical transactional customer data with latent satisfaction and churn behavior.

Note that some variables are used to build authors but will be in practice not available and hidden for the analyst.

We assume $n$ customers at the beginning of the period. We wish to analyze their purchasing data for $T$ periods.

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

Customer satisfaction depends on the number of purchases and can be equally positive or negative. Raw satisfaction is defined as $N_i$ times a uniform variable:

$$
S^{(\text{raw})}_i \mid N_i = n_i \sim n_i \times \mathcal{U}([0,5]).
$$

where $\mathcal{U}$ denotes the (continuous) uniform distribution.

---

### Yearly satisfaction

Yearly satisfaction is capped at 5 (and is greater or equal than 0):

$$
S_i = \min\left( \{ S^{(\text{raw})}_i, 5 \} \right).
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
T_{ij} \mid N_i \sim \mathcal{U}\left(\{ 1,\dots,n_{\text{days}} \} \right),
\quad j \in \{1,\dots,N_i\},
$$

where $n_{\text{days}} \in \{365,366\}$ depending on whether the year is leap or not.

- **(H3)** Purchases are uniformly distributed over the year (no seasonality, holiday, or weekend effects).

---

### Churn

A customer may churn at the end of the year, with a probability that increases when satisfaction is low:

$$
\text{churn}_i \sim \mathcal{B}\left(\frac{1}{1 + 8 \dot S_i}\right),
$$

where $\mathcal{B}$ denotes the Bernoulli distribution.

---

## Time Indexing

The data is generated independently for each year. If a customer churns in a given year, all their future observations are removed.

We introduce a time index $t$ for all variables except heterogeneity:

$$
N_{it} \;
S^{(\text{raw})}_{it} \;
S_{it} \;
X_{it} \;
\text{tot}_{it} \;
T_{ijt} \;
\text{churn}_{it}.
$$

If $\text{churn}_{it} = 1$, all observations for customer $i$ at time $t+1$ and beyond are deleted.

- **(H4)** All years are independent.

---

## Recency

Recency is a key variable for modeling customer lifetime value.

Remind that:
- $T_{i1t}$ is the first purchase date of customer $i$ in year $t$,
- $T_{iN_{it}t}$ is the last purchase date of customer $i$ in year $t$.

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

---

## Customer Lifetime Value
Simulating $T$ years including the future, allows us to knwo their **true lifetime value** by simply summing all their expenses :

$$
CLV_i = \sum_{t=1}^T tot_{it} 
$$

(H) Customers stay active for maximum $T$ years.

The lifetime value is therefore slightly underestimated and the simulation could be performed longer, although it would add "lifespan complications".

## Final data

Variables such as satisfaction and heterogeneity are not observed by the analyst and are only used for data generation.

We obtain a final table of approximately $n \times T - m$ rows (where $m$ is the number of deleted observations due to churn). For each pair (customer ID, year), we observe the following variables:
- average purchase $X_{it}$
- number of purchases $N_{it}$
- total amount $tot_{it}$
- satisfaction (hidden)
- churn
- last purchase
- first purchase the following year
- first purchase since the customer has arrived
- tenure


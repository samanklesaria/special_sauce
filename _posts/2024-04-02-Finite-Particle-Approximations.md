---
category: inference
---

Say you have a discrete distribution $\pi$ that you want to approximate with a small number of weighted particles. Intuitively, it seems like the the best choice of particles would be the outputs of highest probability under $\pi$, and that the relative weights of these particles should be the same under our approximation as they were under $\pi$. This actually isn’t hard to prove! 

### Minimizing the KL Divergence

Let $q$ be our approximation: a version of $\pi$ with support restricted to $b$ outcomes. We’ll try to minimize $KL[q, \pi]$. (Note that $KL[\pi, q]$ will always be infinite as we do not have $ \pi \ll q$, so using this distance isn't an option). Treat $\pi$ and $q$ as vectors in $\mathbb{R}^n$, and assume without loss of generality that $\pi_1 \geq \pi_2 \geq \pi_3 \dotsc$. 

**Claim 1: $KL[q, \pi]$ is minimized when the nonzero components of $q$ have $q \propto \pi$.**  

*Proof:*
Use Lagrange multipliers. Let $S$ be the support of $q$.  $L(\lambda, q) = \sum_{i \in S} q_i \log (q_i / \pi_i) + \lambda (1 - \langle q, 1 \rangle)$. The minimum will be at a fixed point of the Lagrangian. Differentiating, we find that $\log (q_i/\pi_i) + (q_i/q_i) + \lambda = 0$ for all $i$ in $q$'s support, or $\pi_i / q_i =e^{\lambda + 1}$. As this proportionality constant is the same for all $i$, this confirms that minimizer is proportional to $\pi$ whenever it is nonzero. $\square$ 



**Claim 2: If $q \propto m \odot \pi$ where $\odot$ indicates point-wise multiplication and $m$ is a binary mask with $\|m\|_0 = b$ picking out the support of $q$, then $\arg \min KL[q, \pi]$ is obtained when $m_i = 1$ when $i \leq b$ and $m_i = 0$ otherwise.** 

*Proof*:
Let $P = m^T\pi$. The KL divergence simplifies as follows
$$
\begin{align*}
KL[q, \pi] &= \frac{1}{P}\sum_{i=1}^n \pi_i m_i \left( \log (\pi_i / P) - \log \pi_i\right) \\
&= -\log(P)
\end{align*}
$$
 The $m$ maximizing $m^T\pi$ is clearly the one with its mass on the $b$ highest outcomes in $\pi$. $\square$ 

### Minimizing Maximum Mean Discrepancy

The KL divergence isn’t the only way to measure distance between distributions. In fact, it’s not a particularly flexible way, as it only lets us compare our discrete distribution $q$ with other discrete distributions $\pi$. Instead, we can minimize the “maximum mean discrepancy” or MMD. The big idea is to choose $q$ to make $E_{X \sim q}f(X)$ as close as possible to $E_{X \sim \pi} f(X)$ for all functions $f$ in some space $\mathcal{H}$. 
$$
\text{MMD} = \sup_{f \in \mathcal{H}, \|f\| \leq 1} (E_{X \sim \pi} f(X) - E_{X \sim q} f(X))^2
$$
If we assume that $\mathcal{H}$ is closed under negation, squaring this difference won't change the optimal distribution $q^*$ minimizing the expression, so we can equivalently write this as 
$$
\sup_{f \in \mathcal{H}, \|f\| \leq 1} (E_{X \sim \pi} f(X) - E_{X \sim q} f(X))
$$
If we let $\mathcal{H}$ be a [reproducing kernel Hilbert space](https://web.archive.org/web/https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) with kernel $k$, function evaluation $f(x)$ is the same as taking the inner product $\langle f, k(x, \cdot) \rangle$, and we can write the result as 
$$
\sup_{f \in \mathcal{H}, \|f\| \leq 1} (E_{X \sim \pi} \langle f, k(X, \cdot) \rangle - E_{X \sim q} \langle f, k(X, \cdot) \rangle)
$$
You can read more about kernel mean embeddings like this [here](https://web.archive.org/web/https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions).

Let's assume that $\pi$ is discrete, and we can use $\mathbb{R}^n$ as this Hilbert space. 
Letting $\pi$ and $q$ also indicate vectors of component probabilities, the expression simplifies as
$$
\sup_{f \in \mathbb{R}^n, \|f\| \leq 1} \langle f, E_{X \sim \pi}[X], - E_{X \sim q}[X] \rangle = \|\pi - q\|_2^2 \\
$$
Our problem, therefore, is to project $\pi$ onto the space of all $q$ for which $\|q\|_1 = 1$ (the sum of components is 1) and $\|q\|_0 = b$ (the total number of non-zero components is $b$).

This space is not convex, but we can get around this by fixing a binary mask vector $m$ with 1-norm $b$ and minimizing $\|\pi - m \odot q\|_2^2$ over all unit $q$, where $\odot$ indicates the Hadamard product. This gives us the Lagrangian 
$$
L(\lambda, q) = \|m \odot q\|^2 - 2 \langle m \odot \pi, q \rangle + \lambda (\langle q, m \rangle - 1).
$$
The derivative with respect to $q_i$ is zero when $2m_i q_i - 2m_i \pi_i + \lambda m_i = 0$ from which we can conclude that $q_i = \pi_i + c$ for some normalizing constant $c$ shared by all components in the support.

We can plug in this result about the optimal $q_i$ and solve to find the optimal $m_i$.
Let $S$ be the support of $q$ we get from $m$ and $1$ be the all ones vector. We want to minimize 
$$
\begin{align*}
\text{MMD} &= \|m\pi + \frac{\langle 1-m, \pi \rangle}{b}m - \pi\|^2 \\
&= \sum_i \left( 1_{i \notin S} \pi_i + 1_{i \in S}\frac{\langle 1-m, \pi \rangle}{b}\right)^2 \\
&= (1 - b) + \sum_{i \notin S} \pi_i^2
\end{align*}
$$
Clearly, this quantity is minimized by choosing $S$ to contain the $b$ most likely components of $\pi$. 



### Unbiased MMD Minimization

The approximation schemes we've looked at so far have been deterministic: to make a $b$ particle approximation $q$ to a more complicated distribution $\pi$, we always choose $q$'s support $S$ to be the $b$ most likely elements in $\pi$.
But because of this determinism, these approximations are biased. By *biased* here, I mean that it is not generally the case that $E[\sum_{i=1}^n q_i f(x_i)] \neq E_{X \sim \pi} f(X)$ for any function $f$ unless $b$ is large enough to capture the full support of $\pi$.  

If this property (unbiased-ness) was more important to us than truly minimizing the maximum mean discrepancy, we could instead try to choose $S$ stochastically. Specifically, we could do the following:
- Assign each outcome $i$ in $\pi$'s support a weight $w_i$.
- Sample an outcome with probability proportional to the remaining weights. 
- Put that outcome in $S$ and prevent it from being sampled again.
- Repeat $b$ times. 

With this scheme, the number of times outcome $i$ is included in $S$ is given by [Wallenius' non-central hypergeometric distribution](https://web.archive.org/web/https://en.wikipedia.org/wiki/Wallenius%27_noncentral_hypergeometric_distribution) with success-outcomes parameter $m_1 =1$, total outcomes parameter $N$ being the size of $\pi$'s support, draws parameter $n=b$, and odds parameter $\theta_i = \frac{(n-1)w_i}{1 - w_i}$.
If we know the odds parameter $\theta_i$, we can get back $w_i$ using $\sigma(\log(\theta_i / (n-1)))$ where $\sigma$ is the standard logistic function. 

Once we choose the support set $S$, we can deterministically assign probabilities to each outcome in $q$ to minimize the maximum mean discrepancy from our target distribution $\pi$. 
As shown above, the optimal choice is to set $q_i = \pi_i + c$ where $c$ is the equally distributed un-allocated probability mass $\frac{1}{b} \sum_{j \notin S} \pi_j$. 

Now, we can choose the  $w_i$ for each outcome in such a way that $\pi_i = E[q_i]$, making $q$ *unbiased* unlike the deterministic approximation discussed earlier. Let $s_i = P(i \in S)$ and $r_i = 1 - s_i$. 
$$
\begin{align*}
\pi_i &= E[1_{i \in S}(\pi_i + \frac{1}{b}\sum_j 1_{j \notin S} P_j)] \\
&= s_i(\pi_i + \frac{1}{b}\sum_j \pi_j (1 - s_j)) \\
b(1-s_i)\pi_i &= \sum_j \pi_j (1 - s_j) \\
(b - 1)r_i \pi_i &= \sum_{j \neq i} \pi_j r_i
\end{align*}
$$
This holds for all $i$ simultaneously, so letting $P$ be the matrix with $\pi$ along its diagonal and $1$ be the all ones matrix, we can write the above equation in vectorized form as
$$
\begin{align*}
(b-1)Pr &= (1-I)Pr \\
((b-1)I - 1 + I)Pr &= 0 \\
(bI - 1)Pr &= 0 \\
\end{align*}
$$
We can see that if we find an eigenvector of $(bI - 1)P$ with eigenvalue zero, the entries will indicate the probability mass functions of our our non-central hypergeometric distributions at zero. 

However, as $(bI - 1)$ is nonsingular for nonzero $b$, no such eigenvalue exists!
This means this whole approach is actually impossible.  **We can't randomly choose a support and then optimally pick the weights if we want the result to be unbiased.** We can either choose optimal weights, or we can have an unbiased approximation. Having both at the same time is fundamentally impossible.



### Quasi Monte Carlo Approximation

If we'd prefer to have unbiased-ness to optimal weights, there are other finite particle approximation strategies we could use besides standard Monte Carlo.
One simple approach is to use randomized lattice [Quasi Monte Carlo](https://web.archive.org/web/https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method). The basic idea goes like this:
- Let $u_i = i/b$ for $b$ different particles.  
- Choose a random number $s$ between 0 and 1 and add it to all the $u_i$ mod 1. These are unbiased, evenly dispersed samples from a uniform distribution.
- Now apply the inverse CDF $\Phi$ of $\pi$ to each of these samples. Here, we're assuming an ordering among the possible outcomes; if $\Phi(0.3) = 3$, for example, that would mean that 30% of $\pi$'s probability mass is for outcomes below 3. This is known as inverse transform sampling, and guarantees we'll get unbiased samples from $\pi$! We'll set all the weights to be $1/b$, just as in the standard Monte Carlo case. 

Some connections to think about: when the $\pi$ we're trying to approximate is a the empirical distribution of particles during a round of sequential Monte Carlo, this approximation is known as *systematic resampling*.
If, as is often the case in probabilistic programming, we're not trying to sample from a single categorical distribution $\pi$, but rather a sequence of dependent distributions $\pi^1, \pi^2, \dotsc$, we can sample our uniform grid in the unit hypercube rather than along the unit interval and apply the inverse CDF of $\pi^i$ to the $i$th coordinates.
In multiple dimensions, however, a uniform grid might not be the best idea. If we try to pick the lattice structure in a way that minimizes MMD between $q$ and $\pi$, the best choice of lattice will depend on the class of functions $\mathcal{H}$ we're considering for the MMD. For example, one common RKHS for multidimensional spaces has a kernel that is just the product of the kernels for each dimension. If the kernel for one of the dimensions has a lower bandwidth, it will make sense to pack our grid points more tightly in this dimension.

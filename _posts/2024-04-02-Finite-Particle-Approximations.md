---
category: inference
---

Say you have a discrete distribution $\pi$ that you want to approximate with a small number of weighted particles. Intuitively, it seems like the the best choice of particles would be the outputs of highest probability under $\pi$, and that the relative weights of these particles should be the same under our approximation as they were under $\pi$. This actually isn’t hard to prove! 

Let $q$ be our approximation: a version of $\pi$ with support restricted to $b$ outcomes. We’ll try to minimize $KL[q, \pi]$. Treat $\pi$ and $q$ as vectors in $\mathbb{R}^n$, and assume wlog that $\pi_1 \geq \pi_2 \geq \pi_3 \dotsc$. 

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

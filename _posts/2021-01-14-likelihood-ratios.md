---
excerpt: >
  Say you're trying to maximize a likelihood $p_{\theta}(x)$, but you only have an unnormalized version $\hat{p_{\theta}}$ for which...
category: inference
---

Say you're trying to maximize a likelihood $p_{\theta}(x)$, but you only have an unnormalized version $\hat{p_{\theta}}$ for which $p_{\theta}(x) = \frac{\hat{p_\theta}(x)}{N_\theta}$. How do you pick $\theta$? Well, you can rely on the magic of self normalized importance sampling.

$$
\int \hat{p_{\theta}}(x)dx = N_\theta \\
\int \frac{q(x)}{q(x)} \hat{p_{\theta}}(x)dx = N_\theta \\
E_{q(x)}\frac{\hat{p_{\theta}}(x)}{q(x)}=N_\theta
$$


Take a Monte Carlo estimate of the expectation, and you're good to go. Specifically, you can maximize

$$
\log \frac{\hat{p_{\theta}}(x)}{N_\theta} = \log \hat{p_{\theta}}(x) - \log E_{q(x)}\frac{\hat{p_{\theta}}(x)}{q(x)}
$$

A special case is when $q(x)$ is uniform, where this simplifies to $\log \hat{p_{\theta}}(x) - b\log E_{q(x)}\hat{p_{\theta}}(x)$ for constant $b$. This is just the negative sampling rule from Mikolov's famous skip-gram paper!



## Maximizing Implicit Likelihoods

Okay, that's cool, but what if we don't even have an unnormalized $\hat{p_{\theta}}(x)$ and $q(x)$? What if we just had a simulator $q(x)$ that can spit out samples, but doesn't know anything about densities?

We'd like to minimize the KL divergence of between whatever density our sampler $q$ is capturing and the true data distribution $p$. 

$$
-E_{q(x)}\log \frac{p(x)}{q(x)}
$$

Well, that expression looks familiar. Once again, we need to maximize a likelihood ratio! Only this time, we can't evaluate $q(x)$. Instead, we can notice that $\log p(x)/q(x)$ is just the log odds of a sample $x$ coming from $p$ rather than $q$. And estimating the log odds of some event occurring is easy: just build a discriminative binary classifier! Specifically, let $u(x)=p(x)$ if $y=1$ and $u(x)=q(x)$ if $y=0$. Then

$$
\begin{align*}
\frac{p(x)}{q(x)} &= \frac{u(x \vert y=1)}{u(x \vert y=0)} \\
&= \frac{u(y=1, x)}{u(y=0, x)} \frac{u(y=0)}{u(y=1)} \\
&= \frac{u(y=1 \vert x)}{u(y=0 \vert x)} \frac{u(y=0)}{u(y=1)}
\end{align*}
$$

So now we have two objectives to minimize: the one for the classifier $u(y=1 \vert x)$ and the one for the implicit model $q(x)$.  That's just a GAN!

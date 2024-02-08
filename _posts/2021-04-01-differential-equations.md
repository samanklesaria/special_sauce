---
category: math
---

In my freshman year of college, I took an introductory differential equations class. That was nine years ago. I've forgotten pretty much everything, so I thought I'd review a little, trying to generalize the techniques along the way. I'll use [summation notation]() throughout, and write $\frac{\partial^n}{\partial x^n}$ as $\partial^n_x$.  



## Ordinary Differential Equations

Solving ordinary differential equations is mostly an exercise in linear algebra.  

### Initial Value Problems

Let $u : \mathbb{R} \to \mathbb{R}^n$. It's a vector function of time. In an initial value problem, we know that $u'(t) = Au(t)$ for some linear operator $A$, and we know $u(0)$. If $A$ were diagonal, finding $u(t)$ would be easy. Each element would obey $u_i'(t) = A_{ii} u_i(t)$, which means $u_i(t) = e^{A_{ii}t}u_i(0)$. The trick to these problems, therefore, is to express $u$ in a basis where $A$ *is* diagonal: its eigenvector basis. Say $A$ has eigenvectors $v_i$, and $u(t) =\sum_i c_i(t)v_i$ for some $c_i$. Then $Au(t) = \lambda_i c_i(t) v_i$, so
$$
u(t) = \sum_i e^{\lambda_i t}c_i(0) v_i= e^{At}u(0)
$$
This can also be used for constraints of the form $u''(t) = Au'(t) + Bu(t)$, when we're given $u(0)$ and $u'(0)$. Just expand the system into $\begin{pmatrix} u'' \\ u' \end{pmatrix} = \begin{bmatrix} A & B \\ I & 0 \end{bmatrix} \begin{pmatrix} u' \\ u \end{pmatrix}$ , and we're left with the same form we had before.



### Characteristic Equations

More generally, say your differential equation is described by the following linear system:
$$
\begin{pmatrix}
u'''' \\
u''' \\
u'' \\
u' \\
\end{pmatrix}
=
\begin{pmatrix}
a_1 & a_2 & a_3 & a_4 \\
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
\end{pmatrix}
\begin{pmatrix}
u''' \\
u'' \\
u' \\
u
\end{pmatrix}
$$
Let the matrix in the middle be $M$. We want to find $\lambda$ such that $|M-\lambda I| = 0$. Calculating the determinant with a cofactor expansion looks like this:
$$
(a_1 - \lambda) \begin{vmatrix}
-\lambda & 0 & 0 \\
1 & -\lambda & 0 \\
0 & 1 & -\lambda
\end{vmatrix}
- 
a_2 \begin{vmatrix}
1 & 0 & 0 \\
0 & -\lambda & 0 \\
0 & 1 & -\lambda
\end{vmatrix}
+ \dots
$$
Each of these are cofactors are triangular. The determinant of a triangular matrix is the product of pivots. We get $(a_1 - \lambda)(-\lambda)^3 - a_2 (-\lambda)^2 + a_3(-\lambda) - a_4 =0$. 

This simplifies to the *characteristic equation* $\lambda^4 = a_1 \lambda^3 + a_2 \lambda^2 + a_3 \lambda + a_4$, which can be easily read off the top row of the matrix. 

When differential equations are written in the form $ay’’ + by’ + cy = 0$, this means we’re solving a system like this:
$$
\begin{pmatrix}
y'' \\
y' \\
\end{pmatrix}
=
\begin{pmatrix}
-b/a & -c/a \\
1 & 0 \\
\end{pmatrix}
\begin{pmatrix}
y' \\
y
\end{pmatrix}
$$
We get $(-b/a - \lambda)(-\lambda) + c/a =0$. Multiply by $a$ to get $a\lambda^2 + b\lambda + c = 0$. 



## Partial Differential Equations

With partial differential equations (where we can be differentiating with respect to multiple different variables), the simple finite dimensional vector spaces we’ve been using won’t be as useful. We’ll have to use an infinite dimensional basis. 



### Functions are vectors 

Functions $f : Y \to Z$ form a vector space when $Z$ is a field. For example, we could form a basis from the delta functions for different values of $Y$, or we could use the Fourier basis. It’s also pretty easy to see how they form a field. This means that functions $g : X \to Y \to Z$ are also vector spaces, but over the field of functions in $Y \to Z$.  If we have a function $u(x,y) : (X,Y) \to Z$, therefore, we can express it both as $X \to Y \to Z$ (making it a vector space over functions $Y \to Z$) and as $Y \to X \to Z$ (making it a vector space over functions $X \to Z$). To be more concrete, we can both think about $u(x,y)$ as a linear combination of basis functions $u_i(x)$ where the coefficients $c_i(y)$ depend on $y$, as well as a linear combination of basis functions $u_i(y)$ where the coefficients $c_i(x)$ depend on $x$. In other words, any function $u(x,y)$ can be written as $c_iu_i(x)u_i(y)$. 



### Linear operators have eigenvectors

Say $f : A \to B$ is a linear function and $A$ is a vector space. Then, for any $u \in A$, $f(u) = f(c_iu_i) = c_if(u_i)$, where the $u_i$ are a basis for $A$. Specifically, you can choose $u_i$ to be an eigenvector basis for $f$, so that $f(c_iu_i) = c_i \lambda_i u_i$ for eigenvalues $\lambda_i$. The obvious example of this is where the $A$ is a vector and $f$ is a matrix. But as we know from above, $A$ could also be a function, which would make $f$ an operator: a map from functions to functions, like the derivative. 

You can express functions in multiple bases at the same time. Say $g : A \to B$ as well, with eigenbasis $v_j$ and eigenvalues $\phi_j$. Then we can express each of the $u_i$ which are the eigenbasis for $f$, as a linear combination of $a_{ij} v_j$. In other words, $u =c_i a_{ij} v_j$, and $f(u)= \lambda_i \phi_j c_i a_{ij} v_j$. This is partially useful when $f$ and $g$ are different partial differentiation operators.  



### Eigenfunctions of derivatives are exponentials

If the vector space $A$ we’re looking at consists of functions of a single argument, then when $\frac{d^n}{dx^n}v_i=\lambda^n v_i$,  $v_i(x)=e^{\lambda x}$. This is the basis used the Laplace transform. 

If the vector space contains functions $u(x,y)$ of multiple arguments, when $\frac{\partial^n u_i}{\partial x^n} = \lambda^n u_i$, $u_i(x,y) = e^{\lambda x}u_i(y)$.

For second derivatives, we can also use the Fourier basis $u_i(x,y) = e^{-\mathbf{i} \lambda x} u_i(y)$ . 



### Solving homogeneous equations

Say we know that $A\frac{d^2 u}{dx} + B \frac{du}{dx} + Cu = 0$. We can express $u$ in the derivative’s eigenfunction basis to get $(A\lambda_i^2 + B \lambda_i + C)c_i e^{\lambda_i t}=0$, which simplifies to $A\lambda_i^2 + B\lambda_i + C = 0$ (the “characteristic equation”). This means that the only eigenfunctions that make up $u$ are those with eigenvalues that are the roots of the characteristic equation.

We can solve a system of linear equations to get the coefficients $c_i$. If we know the initial conditions $u(0)$ and $\partial_x u(0)$,  when $x=0$, we get that the sum $c_i = u(0)$, and $c_i \lambda_i = \partial_x u(0)$. If, alternately, we know $u(0)$ and $u(X)$ for some $X$, the second equation in the system becomes $c_ie^{\lambda_i T} = u(T)$, which is still linear in the $c_i$.   



### Separation of Variables

Partial differential equations of the form $\partial^n_x u = k^m \partial^m_y u$ are called “separable”. To find what $u(x,y)$ is, we express it in the basis $c_i u_i(x) u_i(y)$. The differential equation tells us that $c_i u_i(y) \partial^n_x u_i(x) = k^m c_i u_i(x) \partial^m_y u_i(y)$. Considering each component of the sum separately, we get $\frac{\partial^n_x u_i(x)}{u_i(x)} = k^m \frac{\partial^m_y u_i(y)}{u_i(y)}$. As one side of the equation is only a function of $x$ and the other is only a function of $y$, both must equal a constant. Call this $\lambda^n$. We get $\partial_x^n u_i(x) = \lambda^n u_i(x)$, which means $u_i(x)=e^{\lambda x}$. For the other side, $\partial_y^m u_i(y) = \frac{\lambda^m}{k^m} u_i(y)$, which means $u_i(y) = e^{\frac{\lambda y}{k}}$. Together, we find that  $u_i(x,y)=e^{\lambda (x + \frac{\lambda y}{k})}$. 


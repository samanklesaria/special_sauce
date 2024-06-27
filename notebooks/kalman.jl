### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 413bc746-f283-11ee-363b-b9a2b9e4b0af
begin
	import Pkg
	Pkg.activate("../environments/slam")
	Pkg.instantiate()
	using LinearAlgebra, SparseArrays, FillArrays, Rotations, RecursiveArrayTools, ForwardDiff, DiffResults, LazyArrays
	using CairoMakie
end

# ╔═╡ f9c28b2a-61f9-44de-a886-a56446e56659
md"""
# Mapping with Gaussian Conditioning
For a robot to navigate autonomously, it needs to learn the locations of any potential obsticles around it. One of the standard ways to do this is with an algorithm known as EKF-Slam. Slam stands for "simultaneous localization and mapping", as the algorithm must simultaneously find out where the robot is (localization) and where the obstacles are (mapping). The "EKF" part refers to the "extended Kalman filter", which is just a fancy name for Gaussian conditioning with Taylor approximations. The idea goes as follows:

Describe the robot by its position coordinates $u \in \mathbb{R}^2$. Assume it has a sensor that gives noisy measurements $Y$ of the displacement to an obstacle at location $v \in \mathbb{R}^2$. Specifically, assume $Y \sim \mathcal{N}(v - u, \Sigma_2)$.

Let our uncertainty about the locations $u$ and $v$ be jointly be described by the random variable $X \sim \mathcal{N}(\mu, \Sigma_1)$ in $\mathbb{R}^4$. Given the observation $Y$, we'd like to find the posterior distribution over $X$.

Conditioning Gaussians is easier in the natural paramteterization. Instead of describing distributions with a mean $\mu$ and covariance $\Sigma$, we describe them with a precision $\Lambda$ and information vector $\Lambda \mu$. In this parameterization, say $X \sim \mathcal{N}(\Lambda_1, \Lambda_1\mu_1)$ and $Y \sim \mathcal{N}(\Lambda_2, \Lambda_2 (Ax  + b))$ where $A$ is a linear transformation. Then by Bayes' rule, a little algebra tells us that
```math
X | y \sim \mathcal{N}(\Lambda_1\mu + A^T\Lambda_2(y - b), \Lambda_1 + A^T\Lambda_2A)
```

When $X$ describes robot and obstacle locations and $Y$ describes noisy displacement observations as above, we get the posterior natural parameters
```math
(\Sigma_1^{-1} \mu + A^T\Sigma_2^{-1}Y, \Sigma_1^{-1} + A^T\Sigma_2^{-1}A)
```
where $A = \begin{bmatrix} -I & I \end{bmatrix}$.

We also need to update the distribution for $X$ when the robot moves. Say we know that the robot's position was displaced by some $\Delta \sim \mathcal{N}(\delta, \Sigma_3)$. After this displacement

```math
X \sim \mathcal{N}\left(\mu + \begin{bmatrix} \delta \\ 0 \end{bmatrix}, \Sigma_1 + \begin{bmatrix} \Sigma_3 &  \\ & 0 \end{bmatrix}\right)
```

By repeatedly updating our distribution over $X$ in response to observations $Y$ and  movements $\Delta$, we can track the likely position of the robot and the obstacle over time.
"""

# ╔═╡ c17dd4df-91c0-4a52-b1e9-13d952cee2e4
md"""
## Adding a Heading

To make this toy example slightly more realistic, let us also give the robot a heading angle $\theta$. Assume now that the sensor measurement above is rotated $-\theta$ degrees, so that $Y = f(X) + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \Sigma_2)$, $f(x) = R_x(x_2 - x_1)$, $x_1$ and $x_2$ refer to the robot and obstacle coordinate components of $x$ respectively, and $R_x$ is the corresponding rotation matrix. Although $f$ isn't linear, we can use the Taylor approximation of $f$ about $\mu$ to pretend it is.
```math
f(x) = f(\mu + (x - \mu)) \approx f(\mu) + \nabla f(\mu)(x - \mu)
```
This makes $Y$ a linear transformation of $X$, so we can continue to use the standard Gaussian conditioning formula. We get natural parameters

```math
(\Sigma_1^{-1} \mu + J^T\Sigma_2^{-1}(Y - b), \Sigma_1^{-1} + J^T\Sigma_2^{-1}J)
```
where $J = \nabla f(\mu)$ and $b = f(\mu) - J \mu$. This expression naturally extends to handling observations for multiple obstacles.  

This is the EKF-Slam algorithm in a nutshell. With the math out of the way, let's get to some code. 
"""

# ╔═╡ 7151ec42-400d-43e3-9b33-06660a426abe
md"""# Code for Observing
While finding the Jacobian above analytically is easy enough, it's even easier to let Julia do it with Forward mode automatic differentiation.

For ease of notation let $L=\Sigma_2^{-2}$.
"""

# ╔═╡ 7c298844-f4f3-4b93-8bd8-98967280f217
L =  10 * Eye(2,2);

# ╔═╡ 9f6d6e5f-6a16-4db2-b12e-a782c4061f3c
function obs_approx(μ::AbstractVector, ix::Int, y::AbstractVector)
	result = DiffResults.JacobianResult(zeros(2), μ)
	f(μ) = RotMatrix{2}(-μ[3]) * (μ[ix:ix+1] - μ[1:2])
	ForwardDiff.jacobian!(result, f, μ)
	f_μ = DiffResults.value(result)
	J = DiffResults.jacobian(result)
	JL = J' * L
	b = f_μ - J * μ
	JL * (y - b), JL * J
end

# ╔═╡ bf6f1d14-9d0d-4dd0-9f06-340279abb389
md"""
# Code for Moving

For simplicity, let's assume that at each step in time, the robot walks in the direction $(1,1)$ and rotates its heading by $2\pi/100$ radians. 
"""

# ╔═╡ 4952ea8c-99ef-4ea2-a3b7-8a39c87e834e
dμ_1 = [1,1,2*π/100];

# ╔═╡ abfd6a4a-f97d-41d8-b028-0c59cd15758a
md"These updates are noisy however, accounting for any imprecision in our dynamics model. Our uncertainty about the robot's location increases at each step according to the following covariance matrix."

# ╔═╡ 347adacf-b17a-4cf8-89d2-99d5bc95babb
Σ3_1 = Diagonal([1,1,0.02]);

# ╔═╡ 5b93a0ff-be9f-4b05-a826-5df344a09146
N = 4; # The number of obstacles

# ╔═╡ 30828091-2b7c-42f1-8974-635f6e681642
md"We add this to our mean vector..."

# ╔═╡ ddb5dfbc-2f57-498e-bccf-daca15ed054f
dμ = sparsevec([1,2,3], dμ_1, 3 + 2 * N);

# ╔═╡ a1e9fbdf-c1c8-43c5-898e-3c998ffe9021
md"...and this to our covariance matrix..."

# ╔═╡ b920ca03-435c-405f-9a4d-fe313067f4a5
Σ3 = sparse_vcat(sparse_hcat(Σ3_1, Zeros(3,2 * N)), Zeros(2*N, 3 + 2 * N))

# ╔═╡ 4ef24a02-a832-42a0-8cc4-3c7f51b32d42
md"""
... which can be factored as $U \Sigma_{3,1} V$, where
$U = \begin{bmatrix} I \\ 0 \end{bmatrix}$ and $V = \begin{bmatrix} I & 0 \end{bmatrix}$. 
"""

# ╔═╡ bb692af8-6400-490e-83bd-306466e7d9b3
U = Vcat(Diagonal(ones(3)), Zeros(2 * N, 3))

# ╔═╡ 1c631432-efe8-4e95-be26-d8ce41d00d7a
V = SparseMatrixCSC(U')

# ╔═╡ 7f57de48-beca-432c-be88-3e2ab87b64f3
md"""
This is a low rank matrix. And we'll need to invert the resut when we condition on the observations, as described above. So we'll express our update to the covariance matrix $\Sigma_1$ in terms of the Woodbury Matrix identity, which says that $(\Sigma_1 +  U\Sigma_{3,1}V)^{-1} = \Sigma_1^{-1} - \Sigma_1^{-1} U (\Sigma_{3,1}^{-1} + V\Sigma_1^{-1} U)^{-1}V\Sigma_1^{-1}$. Let $\Sigma_1^{-1} = \Lambda$. 
"""

# ╔═╡ a6c2c110-2581-4001-bd49-854a2d3ba6ff
Λ3_1 = inv(Σ3_1);

# ╔═╡ eea2d0c7-bf6e-4d93-b49e-26f0b0b48f42
function update_precision(Λ)
	Λ - Λ * U * (((Λ3_1 + Λ[1:3, 1:3]) \ V) * Λ)
end

# ╔═╡ dce95879-1dad-435b-ab6a-747adc482f81
md"""
# Putting them Together
We want to update the distribution over possible locations $X$ after a single time-step. First, the movement of the robot is accounted for by the dynamics model. Then we condition on the observations of each obstacle $ys$.
"""

# ╔═╡ 54f1d653-0b7d-4108-af18-ac2455730df5
tupsum(x,y) = x .+ y;

# ╔═╡ e544f59e-9c7e-4f19-a372-604be29936f1
function step((μ,Λ), ys)
	μ += dμ
	Λ = update_precision(Λ)
	msg = reduce(tupsum, (obs_approx(μ, 4 + 2 * (i-1), y)
		for (i, y) in enumerate(eachcol(ys))))
	(Λμ, Λ) = (Λ * μ, Λ) .+ msg
	(Λ \ Λμ, Λ)
end

# ╔═╡ 80d27bc1-789f-4a22-a336-2d9220fb1972
md"That's it! We've written the algorithm for EKF-Slam. It remains to see how it performs on fake data."

# ╔═╡ 4be488c5-204d-4839-89f6-b818ce58604b
md"""
# Fake Data
Here we generate the unknown, true trajectory that the robot takes by following our dynamics model.
"""

# ╔═╡ 850783aa-9428-4006-a397-dcb62712820f
K = 100; # Number of steps

# ╔═╡ b0035064-b51f-4d52-b1b0-435a9131badd
true_x1s = [zeros(3) cumsum(repeat(dμ_1, 1, K-1) + sqrt(Σ3_1) * randn(3,K-1), dims=2)];

# ╔═╡ c0d718e1-424a-473f-9b7d-7f86f0ba192e
md"We'll also set up the true but unknown obstacle locations."

# ╔═╡ 75314acc-6066-4cf8-9509-bc73eaabbcf1
true_x2s = 10 * rand(2, N);

# ╔═╡ a151e918-adcd-4cfe-b17e-844ff7a97e27
begin
f = Figure()
ax = f[1, 1] = Axis(f)
plot!(true_x1s[1, :], true_x1s[2, :], label="robot path")
scatter!(true_x2s[1,:], true_x2s[2,:], label="obstacles")
f[1,2] = Legend(f, ax)
f
end

# ╔═╡ a0439371-bfa9-4815-8d78-bce7cafdaba4
md"Imagine the displacement observations the robot sees along its trajectory look like this:"

# ╔═╡ 8652caba-9c83-4e76-aaab-d7c31d25260d
intRootL = sqrt(inv(L));

# ╔═╡ 4566931f-3476-4a16-adee-ae9be720b13d
ys = [(RotMatrix{2}(-x[3]) * (true_x2s .- x[1:2])) + intRootL * randn(2,N) for x in eachcol(true_x1s)];

# ╔═╡ 0df3a0f4-95c3-40d1-98d0-8a7dc0b763ea
begin
plt = plot([y[1,1] for y in ys], [y[2,1] for y in ys])
for i in 2:N
	plot!([y[1,i] for y in ys], [y[2,i] for y in ys])
end
plt
end

# ╔═╡ d3075949-f018-403a-a4d2-c22c700200cf
md"""
# Simulating the Algorithm
When we start tracking the robot, we know its location almost (making these precision terms super high), but we know nothing about the locations of the obstacles, so the rest of the precision terms should be low.
"""

# ╔═╡ 7fb32c45-dd11-4529-b0fb-0a89659364fc
Λ1 = Diagonal(vcat(50 * ones(3), (1 / 50) * ones(2 * N)))

# ╔═╡ 7ad46177-f8cf-4358-bd4a-247ba72ab013
start = (zeros(3 + 2 * N), Matrix(Λ1));

# ╔═╡ 7e29ec89-180f-41a3-b6a9-370a382c60fa
progress = VectorOfArray([p[1][1:2] for p in accumulate(step, ys; init=start)]);

# ╔═╡ 043de0e3-9cd8-496f-9a30-85ebaffc9d57
begin
f2 = Figure()
ax2 = f2[1, 1] = Axis(f2)
plot!(true_x1s[1, :], true_x1s[2, :], label="true robot trajectory")
plot!(progress[1, :], progress[2, :], label="inferred robot trajectory")
f2[1,2] = Legend(f2, ax2)
f2
end

# ╔═╡ 5e779882-07ba-445d-b5c3-21a67511700b
md"As we can see, although the robot's trajectory did not follow our dynamics model exactly thanks to the added noise at each step, conditioning on the observations allowed us to keep track of its approximate position regardless." 

# ╔═╡ Cell order:
# ╟─413bc746-f283-11ee-363b-b9a2b9e4b0af
# ╠═f9c28b2a-61f9-44de-a886-a56446e56659
# ╟─c17dd4df-91c0-4a52-b1e9-13d952cee2e4
# ╟─7151ec42-400d-43e3-9b33-06660a426abe
# ╠═7c298844-f4f3-4b93-8bd8-98967280f217
# ╠═9f6d6e5f-6a16-4db2-b12e-a782c4061f3c
# ╟─bf6f1d14-9d0d-4dd0-9f06-340279abb389
# ╠═4952ea8c-99ef-4ea2-a3b7-8a39c87e834e
# ╟─abfd6a4a-f97d-41d8-b028-0c59cd15758a
# ╠═347adacf-b17a-4cf8-89d2-99d5bc95babb
# ╠═5b93a0ff-be9f-4b05-a826-5df344a09146
# ╟─30828091-2b7c-42f1-8974-635f6e681642
# ╠═ddb5dfbc-2f57-498e-bccf-daca15ed054f
# ╟─a1e9fbdf-c1c8-43c5-898e-3c998ffe9021
# ╟─b920ca03-435c-405f-9a4d-fe313067f4a5
# ╟─4ef24a02-a832-42a0-8cc4-3c7f51b32d42
# ╠═bb692af8-6400-490e-83bd-306466e7d9b3
# ╠═1c631432-efe8-4e95-be26-d8ce41d00d7a
# ╟─7f57de48-beca-432c-be88-3e2ab87b64f3
# ╠═a6c2c110-2581-4001-bd49-854a2d3ba6ff
# ╠═eea2d0c7-bf6e-4d93-b49e-26f0b0b48f42
# ╟─dce95879-1dad-435b-ab6a-747adc482f81
# ╠═e544f59e-9c7e-4f19-a372-604be29936f1
# ╠═54f1d653-0b7d-4108-af18-ac2455730df5
# ╟─80d27bc1-789f-4a22-a336-2d9220fb1972
# ╟─4be488c5-204d-4839-89f6-b818ce58604b
# ╠═850783aa-9428-4006-a397-dcb62712820f
# ╠═b0035064-b51f-4d52-b1b0-435a9131badd
# ╟─c0d718e1-424a-473f-9b7d-7f86f0ba192e
# ╠═75314acc-6066-4cf8-9509-bc73eaabbcf1
# ╠═a151e918-adcd-4cfe-b17e-844ff7a97e27
# ╟─a0439371-bfa9-4815-8d78-bce7cafdaba4
# ╠═8652caba-9c83-4e76-aaab-d7c31d25260d
# ╠═4566931f-3476-4a16-adee-ae9be720b13d
# ╠═0df3a0f4-95c3-40d1-98d0-8a7dc0b763ea
# ╠═d3075949-f018-403a-a4d2-c22c700200cf
# ╠═7fb32c45-dd11-4529-b0fb-0a89659364fc
# ╠═7ad46177-f8cf-4358-bd4a-247ba72ab013
# ╠═7e29ec89-180f-41a3-b6a9-370a382c60fa
# ╠═043de0e3-9cd8-496f-9a30-85ebaffc9d57
# ╟─5e779882-07ba-445d-b5c3-21a67511700b

### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ f1f56e00-f997-11ee-29f1-9d0445ca8c23
begin
	import Pkg
	Pkg.activate("/Users/sam/special_sauce/environments/slam")
	Pkg.instantiate()
	using LinearAlgebra, SparseArrays, RecursiveArrayTools, Distributions, Unzip, ExtendableSparse, ForwardDiff, DiffResults, AppleAccelerate, CairoMakie
end

# ╔═╡ 5a0ac6fd-0e0f-4503-8c34-f19705fb43fa
md"""
# Mapping with Graph Slam

For a robot to navigate autonomously, it needs to learn both its own location, as well as the locations of any potential obsticles around it, given its sensors' observations of the world. We'll create a probabilistic model of our environment and get a MAP estimate of these unknown quantities. 

- Let $x_t \in \mathbb{R}^2$ be the robot's location at time $t$.
- Let $θ_t \in \mathbb{R}$ be the robot's heading angle at time $t$.
- Let $m_i \in \mathbb{R}^2$ be the position of the $i$th obstacle.
- Let $o_{t, i} \in \mathbb{R}^2$ be our robot's noisy observation of its distance at time $t$ to obstacle $i$. 
"""

# ╔═╡ 7016081b-73c3-45d9-93cd-0681952824c3
md"""
# Motion Model
We will assume that the robot advances approximately one step forward (in its local coordiante system) at each timestep. Let $R(\theta)$ be a rotation matrix that converts coordinates in the $\theta$-rotated coordinate frame to global coordinates, so that $p(x_t | x_{t-1}) = \mathcal{N}\left(x_{t-1} + R(\theta_{t-1}) \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \sigma^2_x I\right)$. We also assume that the heading angle increases by approximately $2\pi/100$ radians each step, so that  $p(\theta_t | \theta_{t-1}) = \mathcal{N}(\theta_{t-1} + \frac{2\pi}{100}, \sigma^2_θ)$. Let $y_t = \begin{bmatrix} x_t \\ \theta_t \end{bmatrix}$ combine the location and heading parameters, and $y$ be the vector of all the $y_t$. In the code that follows, Gaussian distributions will be represented by their mean and covariance parameters. For reasons we'll see later on, it will be easiest to express the mean at time $t$ by indexing into $y_{t-1}$. 
"""

# ╔═╡ a35b6243-d48b-4c3f-9661-3a5e96b0d10c
Σ_x = Diagonal(fill(0.05, 2));

# ╔═╡ 7c31cd84-d322-4b19-a682-442fa815e8a7
μ_θ(y, θ) = y[θ:θ] + fill(2π/100, 1);

# ╔═╡ f670410f-7625-4219-bcc6-a24451efe2f1
Σ_θ = 0.008;

# ╔═╡ 9ad6c5a6-784e-42df-9d41-44816a988a1a
rot(t) = [cos(t) -sin(t); sin(t) cos(t)]

# ╔═╡ a29ea7bd-f4e4-4de6-aaa0-dd4809adfcb3
μ_x(y, x, θ) = y[x] + rot(y[θ]) * [1.0,0.0]

# ╔═╡ e880166d-f739-445a-bd4e-9bc538b2d0be
start = zeros(3);

# ╔═╡ a040a504-e679-4c14-bd63-54eb255a6d55
md"# Simulating the Model"

# ╔═╡ 89e90ba1-d2ae-4986-8cf5-b6e4ab9c7e7d
md"Let's simulate a robot trajectory that obeys this motion model for $T$ timesteps. We'll try to reconstruct this trajectory with a maximum likelihood approach in the next section."

# ╔═╡ 836c834f-d398-4726-a10e-11b0bfc71b6a
T = 100

# ╔═╡ 68ca2409-2606-4e10-849a-3451ee73c313
mean_step(y) = [μ_x(y, 1:2, 3); μ_θ(y, 3)]

# ╔═╡ f8c3b8c8-c122-474d-936f-7762d6d39051
md"Without noise, the robot's trajectory would look like this, with the color changing from black to yellow over time."

# ╔═╡ 5d3c15e6-659a-4b44-b96e-71d2b70ebd26
yhat = VectorOfArray(accumulate((y,_)-> mean_step(y), 1:T; init=start));

# ╔═╡ 7b1d7164-ed9f-4f83-be07-ba0050eff490
let f1 = Figure()
	ax1 = f1[1, 1] = Axis(f1)
	p1 = plot!(yhat[1, :], yhat[2, :], color=1:T)
	Colorbar(f1[1,2], p1)
	f1
end

# ╔═╡ 1a371f73-0da4-4ce6-b0ed-3113c0e7403e
md"But as we've assumed some noise at each step, the true trajectory is a little more wiggly."

# ╔═╡ 144cb027-bf93-49aa-9abc-3f6f84d8fca3
step(y) = [rand(MultivariateNormal(μ_x(y, 1:2, 3), Σ_x)); rand(Normal(μ_θ(y, 3)[1], Σ_θ))]

# ╔═╡ df5bb6b0-f306-4a2e-86a8-1d5845e13911
true_y = VectorOfArray(accumulate((y,_)-> step(y), 1:T; init=start));

# ╔═╡ 8b845352-92f2-4824-ade3-7435a1d7c230
let f1 = Figure()
	ax1 = f1[1, 1] = Axis(f1)
	p1 = plot!(true_y[1, :], true_y[2, :], color=1:T)
	Colorbar(f1[1,2], p1)
	f1
end

# ╔═╡ b95f1003-be6d-4b1c-96d1-1e866d6c1305
md"""
# Obstacle Location Prior

We will assume an uninformative prior over the locations of our $k$ potential obstacles $m$:  $m_i \sim \mathcal{N}(0, \Sigma_m)$. For consistency with the earlier mean method defined for positions $x$, the mean function for obstacles will take a dummy argument.
"""

# ╔═╡ c15a690c-d6a9-45ba-b67b-26431c3bb81e
k = 5 # Number of observations

# ╔═╡ 23cb7a13-2fa0-4d86-a4f9-bcad1559b53a
μ_m(_) = fill(10, 2);

# ╔═╡ 4bf996e4-272d-4aa4-b086-714cf5413f05
Σ_m = Diagonal(fill(100, 2));

# ╔═╡ d55d801b-4111-4158-8097-862a03951311
true_m = rand(MultivariateNormal(μ_m(nothing), Σ_m), k);

# ╔═╡ 248db8a8-ed1b-4a60-9620-8aff21e26e04
let f1 = Figure()
	ax1 = f1[1, 1] = Axis(f1)
	p1 = plot!(true_y[1, :], true_y[2, :], color=1:T, label="robot trajectory")
	plot!(true_m[1, :], true_m[2,:], color=:red, label="map points")
	f1[1,2] = Legend(f1, ax1)
	f1
end

# ╔═╡ a709321a-90a2-484a-a7d8-58a073994c64
md"""
Let $z = \begin{bmatrix} y \\ m \end{bmatrix}$ combine the ego and obstacle parameters.
"""

# ╔═╡ 0dae8405-37f9-49ba-8a30-44bce8b87b2f
true_z = [vec(true_y); vec(true_m)];

# ╔═╡ 26378809-c2c2-474c-b673-5f1a0feb7f12
md"""
# Observation Model
Let $p(o_{t, i} | x_t, θ_t, m_i) = \mathcal{N}(R(-θ_t)(m_i - x_t), Σ_o)$, so that the robot's sensor tells it approximately how close each map point is in its local coordinate frame. 
"""

# ╔═╡ 0c53540c-e175-47ac-a0cd-66f126942529
μ_obs(z, x, θ, m) = rot(-z[θ]) * (z[m] - z[x])

# ╔═╡ 7f7b113b-7a5e-46c3-99d7-14dd7e20fe83
Σ_obs = Diagonal(fill(0.1, 2));

# ╔═╡ 1e31e447-01fa-4e90-adf8-1dd9b71190f8
obs = VectorOfArray([VectorOfArray([
	rand(MultivariateNormal(μ_obs(true_z, 3t+1:3t+2, 3t+3, (3T+2m+1):(3T+2m+2)), Σ_obs)) for m in 0:(k-1)]) for t in 0:(T-1)]);

# ╔═╡ a9b3ee42-8429-4a9c-a0d5-061f0ba3a562
md"The following plots our observations over time, from black at the start to yellow at time $T$. The swirling shape comes from the fact that the robot spins as it moves."

# ╔═╡ d82b3280-5813-4a4a-80a2-ae77405e06ce
let f = Figure()
	ax = f[1, 1] = Axis(f)
	for i in 1:k
		p1 = plot!(obs[1,i,:], obs[2, i,:], color=1:T)
		if i == 1
			Colorbar(f[1,2], p1)
		end
	end
	f
end

# ╔═╡ 4db531ef-f57b-49de-af1e-68a935101786
md"""
# Guessing an Initial Trajectory
We'll start out with our guess $\hat{y}$ as the mean of the motion model. We'll guess $\hat{m}$ by just sampling from the prior.
"""

# ╔═╡ 828dbe68-902f-4d13-a640-9fcfdb6ebbda
mhat = rand(MultivariateNormal(μ_m(nothing), Σ_m), k);

# ╔═╡ 10c1f9b3-494b-4a4c-b417-d3000fffb7ae
zhat = [vec(yhat); vec(mhat)];

# ╔═╡ 8d2d8877-eafd-48bd-8b75-d588b38d3ff4
md"""
# Assembling Trajectory Probability
The negative joint log density of our guess and observations $L(y, m)$ is a sum of factors $L_i = (v_i - μ_i(z))^TΛ_i(v_i - μ_i(z))$ for different variables $v_i$ with means $\mu_i$ and precisions $\Lambda_i$. The following code assembles these factors for the observation model above.
"""

# ╔═╡ def9e3a1-0d0c-490e-a707-3bbbcd7ef191
start_x(_) = [0., 1.0]

# ╔═╡ 09af1db3-19cb-4059-8950-77a33d59df85
start_θ(_) = fill(2π/100, 1);

# ╔═╡ 16ea1c4c-2f9a-47c7-993d-7b748d2fa8b4
md"""
# Maximizing Trajectory Probability
We know $L_i(z) = -f_i(z)^T\Lambda_i f(z)$ where $f_i$ is a potentially nonlinear transformation of $z$. Taylor expand $f_i(z) \approx f_i(\hat{z}) + J_i\Delta$ where $\Delta = z - \hat{z}$ and $J_i = \nabla_z f_i(\hat{z})$. Substitute this expression in definition of $L_i$ giving $L_i(z) \approx \Delta^T H_i\Delta + 2 b_i^T\Delta + c$, where $H_i = J_i^T\Lambda_i J_i$ and $b_i = J_i^T\Lambda_i f_i(\hat{z})$.
When add up all these factors to get the full log probability, we get $\Delta^T H \Delta + 2 b^T \Delta + c$ where $b$ is the sum of the $b_i$ and $H$ is the sum of the $H_i$. 

To minimize this quantity, take the derivative. We find that $L$ will be approximately minimized when $H\Delta = -b$ or $\Delta = -H^{-1}b$. It remains to solve for $\Delta$ and modify $z$ appropriately. Repeating this gives the Gauss Newton algorithm.

When $f$ is not well approximated by its first order Taylor expansion, instead of solving $H^{-1}b$, it works better to solve the smoothed version $(H + \lambda \text{Diag}(H))^{-1}b$, where $\lambda$ is a factor that gets slowly lowered to zero as $z$ nears its optimal value. This is the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm) algorithm.
"""

# ╔═╡ 31432aa9-862d-445e-8de3-90e621744dbb
md"""
### Understanding `log_prob_builder`
The `log_prob_builder` function above has two forms, depending on its `jac` argument. If `jac=true`, the function builds a vector of terms $H_i$ and $b_i$ described above. Othewise, it builds a vector of the negative log probabilities $L_i$.  
"""

# ╔═╡ 312a2254-3319-48e3-b764-d0e24068c86c
const QuadformBuilder = Vector{Tuple{<: ExtendableSparseMatrix, Vector}};

# ╔═╡ 445f55dd-76c0-4fc5-8cf2-881091375767
md"These terms are computed by the `factor` function, which assembles $H$ and $b$ out of $f_i(z)$ and its jacobian $J$."

# ╔═╡ b48ffaab-2f05-4abd-a844-2a38014eee04
factor(Λ, (fval,_)::Tuple{<:Any, Nothing}) = fval' * (Λ * fval)

# ╔═╡ 013405ea-720f-4564-840a-9d491edbf4e3
function factor(Λ, (fval, J))
	ΛJ = Λ * J
	b = (ΛJ)' * fval
	H = J' * ΛJ
	(ExtendableSparseMatrix(H), b)
end

# ╔═╡ 9336d090-6069-48bf-ba8c-43f3f5a680db
md"Finally, we compute $f_i(z)$ and its Jacobian using the following wrappers around the `sparse_jac` function, which computes a function and its sparse Jacobian."

# ╔═╡ 439e3e19-e843-481e-9be4-3145a06d705f
md"""
# Handling Sparsity
To define the `sparse_jac` function, we'll wrap the `jacobian` function from the `ForwardDiff` library. 
"""

# ╔═╡ dad28db4-7546-40fa-a9d5-ac0af30dea66
function sparse_jac(f, z, support, outs; jac=true)
	jac || return f(z), nothing
	M = nothing
	fz = nothing
	function f_wrapper(z_sup)
		newz = collect(eltype(z_sup), z)
		newz[support] .= z_sup
		f(newz)
	end
	z_sup = z[support]
	res = DiffResults.JacobianResult(zeros(outs), z_sup)
	res = ForwardDiff.jacobian!(res, f_wrapper, z_sup)
	J = DiffResults.jacobian(res)
	M = ExtendableSparseMatrix(outs, length(z))
	M[:, support] .= J
	DiffResults.value(res), M
end

# ╔═╡ 3d6faafc-c396-43ad-998d-8f3d8ed0a556
function term(z, target, μ, outs, ixs...; jac=true)
	support = [target; reduce(vcat, ixs; init=Int[])]
	f(z) = z[target] - μ(z, ixs...)
	sparse_jac(f, z, support, outs; jac)
end

# ╔═╡ 27c3ce11-1ff7-46c4-9b55-5a3c63481692
function obs_term(z, obs, ixs...; jac=true)
	support = reduce(vcat, ixs; init=Int[])
	f(z) = obs - μ_obs(z, ixs...)
	sparse_jac(f, z, support, 2; jac)
end

# ╔═╡ 970dbc27-31a0-412e-81d4-367f4c05b86c
function log_prob_builder(z; jac=true)
	bld = jac ? QuadformBuilder() : Vector{Float64}()
	
	# We know the location distribution at time 1
	push!(bld, factor(inv(Σ_x), term(z, 1:2, start_x, 2; jac)))
	push!(bld, factor(inv(Σ_θ), term(z, 3:3, start_θ, 1; jac)))
	
	# Add the Markovian jump probabilities at each step
	for t in 1:(T-1)
		ix = 3t+1
		pix = 3(t-1)+1
		push!(bld, factor(inv(Σ_x), term(z, ix:ix+1, μ_x, 2, pix:pix+1, pix+2; jac)))
		push!(bld, factor(inv(Σ_θ), term(z, ix+2:ix+2, μ_θ, 1, pix+2; jac)))
	end
	
	# Add the prior on map components
	for i in 0:(k-1)
		ix = 3T+2i+1
		push!(bld, factor(inv(Σ_m), term(z, ix:(ix+1), μ_m, 2; jac)))
	end

	# Add the observations
	for t in 0:(T-1)
		for i in 1:k
			m = 3T+2(i-1)
			ix = 3t+1
			push!(bld, factor(inv(Σ_obs), obs_term(z, obs[:, i, t+1], ix:(ix+1), ix+2,(m+1):(m+2); jac)))
		end
	end
	jac ? sum.(unzip(bld)) : sum(bld)
end

# ╔═╡ f74f5569-fd45-492a-bafa-e59d19e4f93a
function maximize_logprob(z; λ=1e-3, α=2, β=3, maxiter=100, eps=1e-4)
	Δ = ones(length(z))
	i = 0
	prevL = log_prob_builder(z; jac=false)
	L = 0.0
	while any(abs.(Δ) .> eps)  && i < maxiter
		H, b = log_prob_builder(z; jac=true)
		while true
			Δ = (H + λ * Diagonal(H)) \ b
			L = log_prob_builder(z - Δ; jac=false)
			L >= prevL || break
			λ *= β
		end
		z[:] .-= Δ
		λ /= α
		prevL = L
		i += 1
	end
	println("Concluded after $i iterations")
	z
end

# ╔═╡ 65601b91-c2f3-43f7-bdca-047b5bcdedd7
md"# Running the Optimization"

# ╔═╡ a24ed4e6-6eae-4944-80e5-005093beaccd
md"Here's the negative log probability of the robot's true trajectory."

# ╔═╡ a9705df9-d39e-48db-a3b7-e906f14367ba
log_prob_builder(true_z; jac=false)

# ╔═╡ 0932d9f2-7bb7-42f4-81c2-970c0e6e9c43
md"Wereas here's how likely our prior mean would be:"

# ╔═╡ 6882d2c7-f778-4f79-9388-7a3ee28dedd5
log_prob_builder(zhat; jac=false)

# ╔═╡ 31eb6da0-78a4-4709-b7f1-bdb7fbfe7284
md"Running Levenberg-Marquardt gives us a solution that isn't our true trajectory, but is technically more likely under the generative model above. "

# ╔═╡ c4f7601c-395e-48bc-afca-b083c99b6c18
z_guess = maximize_logprob(copy(zhat));

# ╔═╡ fd5cdd3f-8490-48b4-b0e3-79054b2ae0eb
log_prob_builder(z_guess; jac=false)

# ╔═╡ 237a90fb-a570-416c-948b-1ca16c657c1a
md"Below, we plot the guessed trajectory as a line, with the true trajectory represented as a scatter plot."

# ╔═╡ bdfb3d49-d39e-4968-8a6a-53fec23cf09e
y_guess = reshape(z_guess[1:3T], (3, T));

# ╔═╡ 61b97d81-0b04-4ee5-8dcd-356f5724827d
let f1 = Figure()
	ax1 = f1[1, 1] = Axis(f1)
	p1 = plot!(true_y[1, :], true_y[2, :], color=1:T)
	lines!(y_guess[1,:], y_guess[2,:], color=1:T+1)
	Colorbar(f1[1,2], p1)
	f1
end

# ╔═╡ dbf2c660-20aa-4d3b-b1c0-2765d0439a97
md"We ended up with a pretty good estiamte of the map points' locations as well."

# ╔═╡ 72494e77-341a-4fe9-9721-5fe62b8f9f79
m_guess = reshape(z_guess[3T+1:end], (2, k));

# ╔═╡ fe99dba8-e39d-4437-9749-2655a6a49a93
let f1 = Figure()
	ax1 = f1[1, 1] = Axis(f1)
	plot!(true_m[1, :], true_m[2,:], color=:red, label="map points")
	plot!(m_guess[1, :], m_guess[2,:], color=:black, label="guesses")
	f1[1,2] = Legend(f1, ax1)
	f1
end

# ╔═╡ Cell order:
# ╠═f1f56e00-f997-11ee-29f1-9d0445ca8c23
# ╟─5a0ac6fd-0e0f-4503-8c34-f19705fb43fa
# ╟─7016081b-73c3-45d9-93cd-0681952824c3
# ╠═a29ea7bd-f4e4-4de6-aaa0-dd4809adfcb3
# ╠═a35b6243-d48b-4c3f-9661-3a5e96b0d10c
# ╠═7c31cd84-d322-4b19-a682-442fa815e8a7
# ╠═f670410f-7625-4219-bcc6-a24451efe2f1
# ╠═9ad6c5a6-784e-42df-9d41-44816a988a1a
# ╠═e880166d-f739-445a-bd4e-9bc538b2d0be
# ╟─a040a504-e679-4c14-bd63-54eb255a6d55
# ╟─89e90ba1-d2ae-4986-8cf5-b6e4ab9c7e7d
# ╠═836c834f-d398-4726-a10e-11b0bfc71b6a
# ╠═68ca2409-2606-4e10-849a-3451ee73c313
# ╟─f8c3b8c8-c122-474d-936f-7762d6d39051
# ╠═5d3c15e6-659a-4b44-b96e-71d2b70ebd26
# ╟─7b1d7164-ed9f-4f83-be07-ba0050eff490
# ╟─1a371f73-0da4-4ce6-b0ed-3113c0e7403e
# ╠═144cb027-bf93-49aa-9abc-3f6f84d8fca3
# ╟─8b845352-92f2-4824-ade3-7435a1d7c230
# ╠═df5bb6b0-f306-4a2e-86a8-1d5845e13911
# ╟─b95f1003-be6d-4b1c-96d1-1e866d6c1305
# ╠═c15a690c-d6a9-45ba-b67b-26431c3bb81e
# ╠═23cb7a13-2fa0-4d86-a4f9-bcad1559b53a
# ╠═4bf996e4-272d-4aa4-b086-714cf5413f05
# ╠═d55d801b-4111-4158-8097-862a03951311
# ╟─248db8a8-ed1b-4a60-9620-8aff21e26e04
# ╟─a709321a-90a2-484a-a7d8-58a073994c64
# ╠═0dae8405-37f9-49ba-8a30-44bce8b87b2f
# ╟─26378809-c2c2-474c-b673-5f1a0feb7f12
# ╠═0c53540c-e175-47ac-a0cd-66f126942529
# ╠═7f7b113b-7a5e-46c3-99d7-14dd7e20fe83
# ╠═1e31e447-01fa-4e90-adf8-1dd9b71190f8
# ╟─a9b3ee42-8429-4a9c-a0d5-061f0ba3a562
# ╟─d82b3280-5813-4a4a-80a2-ae77405e06ce
# ╟─4db531ef-f57b-49de-af1e-68a935101786
# ╠═828dbe68-902f-4d13-a640-9fcfdb6ebbda
# ╠═10c1f9b3-494b-4a4c-b417-d3000fffb7ae
# ╟─8d2d8877-eafd-48bd-8b75-d588b38d3ff4
# ╠═970dbc27-31a0-412e-81d4-367f4c05b86c
# ╠═def9e3a1-0d0c-490e-a707-3bbbcd7ef191
# ╠═09af1db3-19cb-4059-8950-77a33d59df85
# ╟─16ea1c4c-2f9a-47c7-993d-7b748d2fa8b4
# ╠═f74f5569-fd45-492a-bafa-e59d19e4f93a
# ╟─31432aa9-862d-445e-8de3-90e621744dbb
# ╠═312a2254-3319-48e3-b764-d0e24068c86c
# ╟─445f55dd-76c0-4fc5-8cf2-881091375767
# ╠═b48ffaab-2f05-4abd-a844-2a38014eee04
# ╠═013405ea-720f-4564-840a-9d491edbf4e3
# ╟─9336d090-6069-48bf-ba8c-43f3f5a680db
# ╠═3d6faafc-c396-43ad-998d-8f3d8ed0a556
# ╠═27c3ce11-1ff7-46c4-9b55-5a3c63481692
# ╟─439e3e19-e843-481e-9be4-3145a06d705f
# ╠═dad28db4-7546-40fa-a9d5-ac0af30dea66
# ╟─65601b91-c2f3-43f7-bdca-047b5bcdedd7
# ╟─a24ed4e6-6eae-4944-80e5-005093beaccd
# ╠═a9705df9-d39e-48db-a3b7-e906f14367ba
# ╟─0932d9f2-7bb7-42f4-81c2-970c0e6e9c43
# ╠═6882d2c7-f778-4f79-9388-7a3ee28dedd5
# ╟─31eb6da0-78a4-4709-b7f1-bdb7fbfe7284
# ╠═c4f7601c-395e-48bc-afca-b083c99b6c18
# ╠═fd5cdd3f-8490-48b4-b0e3-79054b2ae0eb
# ╟─237a90fb-a570-416c-948b-1ca16c657c1a
# ╠═bdfb3d49-d39e-4968-8a6a-53fec23cf09e
# ╟─61b97d81-0b04-4ee5-8dcd-356f5724827d
# ╟─dbf2c660-20aa-4d3b-b1c0-2765d0439a97
# ╠═72494e77-341a-4fe9-9721-5fe62b8f9f79
# ╟─fe99dba8-e39d-4437-9749-2655a6a49a93

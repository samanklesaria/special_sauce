### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ e1e011e8-3e91-4317-a081-9a23f39349c8
using KernelFunctions,LinearAlgebra, AbstractGPs, ArraysOfArrays

# ╔═╡ 4140ca9d-2530-43a9-aee8-73f7fb0770ec
using BenchmarkTools

# ╔═╡ b18c628c-cced-11ee-0033-51bdcc63c29c
md"""
## Finite Basis Gaussian Processes
By Mercer's theorem, every positive definite kernel $k(x, y) : \mathcal{X} \to \mathcal{X} \to \mathbb{R}$ that we might want to use in a Gaussian Process corresponds to some inner product $\langle \phi(x), \phi(y) \rangle$, where $\phi : \mathcal{X} \to \mathcal{V}$ maps our inputs into some other space.  For many kernels (like the venerable RBF), this space is infinite dimensional, and we can't work with it directly. But when it's finite dimensional (in say $d$ dimensions), we can! This lets us avoid the usual $O(n^3)$ scaling for Gaussian process regression, getting $O(nd+d^3)$ instead. 
"""

# ╔═╡ cd5118fc-1129-467e-be5e-9f786fa114f3
begin
struct FiniteBasis{T} <: Kernel
	ϕ::T
end
(k::FiniteBasis)(x, y) = dot(k.ϕ(x), k.ϕ(y))
end

# ╔═╡ 8c973260-b0a0-49fe-9b18-7dd8ee6d467b
md"""
We will use the *weight space* view of Gaussian processes, which interprets GP regression as Bayesian linear regression. We assume that there is a weight vector $w : \mathcal{V}$ with prior $\mathcal{N}(0, I)$, and that $y \sim \mathcal{N}(X w, I)$, where $X$ is the matrix for which row $i$ is given by $\phi(x_i)$. 
The posterior over $w$ remains Gaussian with precision $\Lambda = I + X^T X$ and mean $\mu = \Lambda^{-1} X^T y$. To make a prediction at $x_*$, we simply find $\langle \phi(x_*), w \rangle$. 
"""

# ╔═╡ 8770765c-e519-4cd7-9eed-da2b50190895
md"""
On the face of it, this seems like a very different generative model than the traditional depiction of Gaussian processes in which the observations $y$ are noisy versions of the function values $f$, which are all jointly Gaussian with a covariance matrix given by the associated kernel. But with a little algebra, one can show that the posterior over $f(x_*) = \langle \phi(x_*), w \rangle$ in the weight space view is the same as the posterior over $f(x_*)$ is the traditional function-space view. 

First, we can marginalize out $w$ to find that

```math
f(x_*) | y \sim \mathcal{N}(X_* \mu, X_* \Lambda^{-1} X_*^T)
```
The mean expands to $X_*(I + X^T X)^{-1} X^T y$ and the variance expands to 
$X_*(I + X^T X)^{-1}X_*^T$.


Now, we can use the Woodbury Matrix Identity, which says that
```math
(I + X^TX)^{-1} = I - X^T(I + XX^T)^{-1}X
```
This lets the mean simplify to
$X_*X^T (XX^T + I)^{-1}y$ and the variance simplify to $X_*X_*^T -X_*X^T(XX^T + I)^{-1}XX_*^T$. Letting $XX^T = K$, we recover the familiar function space representation of Gaussian process. See the first chapter of the [Rasmussen book](http://gaussianprocess.org/gpml/) for a more detailed derivation. 
"""

# ╔═╡ ca1fa3bc-c6a8-400a-93ed-850821f57b1f
import AbstractGPs: AbstractGP, FiniteGP

# ╔═╡ 38ccb850-2091-4914-a7fa-0fdfe0e64375
struct DegeneratePosterior{P,T,C} <: AbstractGP
	prior::P
	w_mean::T
	w_prec::C
end

# ╔═╡ 2646ce1f-bd5c-4ea0-a9fb-c6f786984240
weight_form(ϕ, x) = flatview(ArrayOfSimilarArrays(ϕ.(x)))'

# ╔═╡ 3a04e978-770c-45d1-be1c-5cc1c1925eb2
import Statistics

# ╔═╡ 8c2d7c8b-8d9e-44fe-88b9-00fd3e3aee14
function Statistics.mean(f::DegeneratePosterior, x::AbstractVector)
	w = f.w_mean
	X = weight_form(f.prior.kernel.ϕ, x)
	X * w
end

# ╔═╡ 841129b7-0d84-4f6b-9ae0-edf2a6cb8661
function AbstractGPs.posterior(fx::FiniteGP{GP{M, B}}, y::AbstractVector{<:Real}) where {M, B <: FiniteBasis}
	kern = fx.f.kernel
	δ = y - mean(fx)
	X = weight_form(kern.ϕ, fx.x)
	X_prec = X' * inv(fx.Σy)
	Λμ = X_prec * y
	prec = cholesky(I + Symmetric(X_prec * X))
	w = prec \ Λμ
	DegeneratePosterior(fx.f, w, prec)
end

# ╔═╡ 9fe52847-ae03-461e-a5f9-fb95adc63cb4
function Statistics.cov(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(f.prior.kernel.ϕ, x)
	AbstractGPs.Xt_invA_X(f.w_prec, X')
end

# ╔═╡ 0f937650-77db-43cb-859a-a0f27dcc464d
function Statistics.var(f::DegeneratePosterior, x::AbstractVector)
	X = weight_form(f.prior.kernel.ϕ, x)
	AbstractGPs.diag_Xt_invA_X(f.w_prec, X')
end

# ╔═╡ 04f817b3-ac62-4597-9969-1232cb416739
md"We can compare the results of this optimized implementation with the standard posterior implementation to ensure that the two agree on the output."

# ╔═╡ 6676671e-e5a6-46d7-b2be-ac56a1038b77
x = rand(2, 2000);

# ╔═╡ 837d7b11-5517-41bc-af12-6dc839645701
y = sin.(norm.(eachcol(x)));

# ╔═╡ 53bcaf58-abbf-4017-9f61-5e22513d4214
kern = FiniteBasis(identity);

# ╔═╡ 87e43518-5c04-4a8a-8f32-10ff1dbff759
f = GP(kern)

# ╔═╡ 55a55d3b-0756-4078-ad2e-aa5a728638ee
fx = f(x, 0.001);

# ╔═╡ 57956b60-9623-42bd-9771-52b4e8d768bc
x2 = ColVecs(rand(2, 2000));

# ╔═╡ 4429526e-8639-47d5-9c67-4844fd38eabc
opt_m, opt_C = @btime mean_and_cov(posterior($fx, $y)($x2))

# ╔═╡ e3ceb531-9ccf-4b1c-a1f3-88d2a6fbbda9
md"To compare against the implementation that uses a function-space perspective, we'll use a bit of a hack: by adding a `ZeroKernel` to our `FiniteBasis` kernel, we get a kernel for which our custom `posterior` method won't be called."

# ╔═╡ c195a0d2-de53-4f9d-962e-56642e4cd01a
fx2 = GP(kern + ZeroKernel())(x, 0.001);

# ╔═╡ 84638922-0fd7-495f-bb2f-a4799596432f
m, C = @btime mean_and_cov(posterior($fx2, $y)($x2));

# ╔═╡ df24f649-f85d-41da-9c35-92eb4a723510
maximum(abs.(opt_C .- C)), maximum(abs.(opt_m .- m))

# ╔═╡ 4e647fbe-f05e-409a-be3b-0ce2b7806aa5
md"Our optimized technique produces the same results!"

# ╔═╡ 2890892f-b8dd-4784-a273-a4aa79549523
md"""
## Random Fourier Features
One application of this technique is the *Random Fourier Features* approximation. By Bochner's theorem, every kernel of the form $k(x,y) = f(x-y)$ for some $f$ can be expressed in the Fourier basis as $f(x-y) = E e^{i\omega (x-y)}$, where the distribution from which $\omega$ is sampled determines the kernel. A Monte Carlo estimate of this expectation is just $\sum_{w_j} e^{i w_j x}e^{-i w_j y}$, which is an inner product of features of the form $\phi_j(x) = e^{i w_j x}$. With some algebraic simplifications (see [here](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/#a4-alternative-random-fourier-features) for a good derivation) we can ignore the imaginary parts and express this as $\phi_j(x)=(\cos(w_j x), \sin(w_j x))$. 
"""

# ╔═╡ 3c9277e0-64d0-4c7e-a416-34b5f71e2056
begin
struct RandomFourierFeature
	ws::Vector{Float64}
end
RandomFourierFeature(kern::SqExponentialKernel, k::Int) = RandomFourierFeature(randn(k))
function (f::RandomFourierFeature)(x)
	Float64[cos.(f.ws .* x); sin.(f.ws .* x)] .* sqrt(2/length(f.ws))
end
end

# ╔═╡ f2e36526-aa4e-41bd-88ea-2bf5b172c1c4
md"To support other spectral densities besides the RBF, we could add constructors for `RandomFourierFeature`."

# ╔═╡ 6b42b8a3-81c8-4dc6-a640-c2b9a78de284
rbf = SqExponentialKernel();

# ╔═╡ 20c58a01-f98d-4cf0-9575-e544f605fe1e
flat_x = rand(2000);

# ╔═╡ 36b6d708-7a06-4954-8dd7-4d49278057b7
flat_x2 = rand(100);

# ╔═╡ 9529f9d1-a905-40e3-bc68-428d85d24fcd
ffkern = FiniteBasis(RandomFourierFeature(rbf, 100));

# ╔═╡ a2df8f7e-9077-4150-b4dc-322d3b9591db
ff_m, ff_C = mean_and_cov(posterior(GP(ffkern)(flat_x, 0.001), y)(flat_x2));

# ╔═╡ 608c630e-6af9-4199-8121-72cd1eea0c6b
m2, C2 = mean_and_cov(posterior(GP(rbf)(flat_x, 0.001), y)(flat_x2));

# ╔═╡ 573de207-995d-46b4-bd4d-dd792b20924c
maximum(abs.(m2 .- ff_m)), maximum(abs.(C2 .- ff_C))

# ╔═╡ 2180ee59-779b-491c-ae8b-b87f2f4eb530
md"Even with only 100 samples, we get a pretty close approximation!"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractGPs = "99985d1d-32ba-4be9-9821-2ec096f28918"
ArraysOfArrays = "65a8f2f4-9b39-5baf-92e2-a9cc46fdf018"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
KernelFunctions = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractGPs = "~0.5.19"
ArraysOfArrays = "~0.6.4"
BenchmarkTools = "~1.4.0"
KernelFunctions = "~0.10.63"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.3"
manifest_format = "2.0"
project_hash = "e3677c9dbe6e4b0491e96402c6c2b131808fe07e"

[[deps.AbstractGPs]]
deps = ["ChainRulesCore", "Distributions", "FillArrays", "IrrationalConstants", "KernelFunctions", "LinearAlgebra", "PDMats", "Random", "RecipesBase", "Reexport", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "6e5e13c57dbfdedddbc3ef727586d8ee0703d50a"
uuid = "99985d1d-32ba-4be9-9821-2ec096f28918"
version = "0.5.19"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArraysOfArrays]]
deps = ["Statistics"]
git-tree-sha1 = "5cf8e9553795d8ceb14a75059389b1a21ab70565"
uuid = "65a8f2f4-9b39-5baf-92e2-a9cc46fdf018"
version = "0.6.4"

    [deps.ArraysOfArrays.extensions]
    ArraysOfArraysAdaptExt = "Adapt"
    ArraysOfArraysChainRulesCoreExt = "ChainRulesCore"
    ArraysOfArraysStaticArraysCoreExt = "StaticArraysCore"

    [deps.ArraysOfArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "ad25e7d21ce10e01de973cdc68ad0f850a953c52"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.21.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "166c544477f97bbadc7179ede1c1868e0e9b426b"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.7"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.KernelFunctions]]
deps = ["ChainRulesCore", "Compat", "CompositionsBase", "Distances", "FillArrays", "Functors", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Random", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "TensorCore", "Test", "ZygoteRules"]
git-tree-sha1 = "654dead9bd3313311b61087ab7a0cf2b7a935cb1"
uuid = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
version = "0.10.63"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─b18c628c-cced-11ee-0033-51bdcc63c29c
# ╠═e1e011e8-3e91-4317-a081-9a23f39349c8
# ╠═cd5118fc-1129-467e-be5e-9f786fa114f3
# ╟─8c973260-b0a0-49fe-9b18-7dd8ee6d467b
# ╟─8770765c-e519-4cd7-9eed-da2b50190895
# ╠═ca1fa3bc-c6a8-400a-93ed-850821f57b1f
# ╠═38ccb850-2091-4914-a7fa-0fdfe0e64375
# ╠═2646ce1f-bd5c-4ea0-a9fb-c6f786984240
# ╠═841129b7-0d84-4f6b-9ae0-edf2a6cb8661
# ╠═3a04e978-770c-45d1-be1c-5cc1c1925eb2
# ╠═8c2d7c8b-8d9e-44fe-88b9-00fd3e3aee14
# ╠═9fe52847-ae03-461e-a5f9-fb95adc63cb4
# ╠═0f937650-77db-43cb-859a-a0f27dcc464d
# ╟─04f817b3-ac62-4597-9969-1232cb416739
# ╠═6676671e-e5a6-46d7-b2be-ac56a1038b77
# ╠═837d7b11-5517-41bc-af12-6dc839645701
# ╠═53bcaf58-abbf-4017-9f61-5e22513d4214
# ╠═87e43518-5c04-4a8a-8f32-10ff1dbff759
# ╠═55a55d3b-0756-4078-ad2e-aa5a728638ee
# ╠═57956b60-9623-42bd-9771-52b4e8d768bc
# ╠═4140ca9d-2530-43a9-aee8-73f7fb0770ec
# ╠═4429526e-8639-47d5-9c67-4844fd38eabc
# ╟─e3ceb531-9ccf-4b1c-a1f3-88d2a6fbbda9
# ╠═c195a0d2-de53-4f9d-962e-56642e4cd01a
# ╠═84638922-0fd7-495f-bb2f-a4799596432f
# ╠═df24f649-f85d-41da-9c35-92eb4a723510
# ╟─4e647fbe-f05e-409a-be3b-0ce2b7806aa5
# ╟─2890892f-b8dd-4784-a273-a4aa79549523
# ╠═3c9277e0-64d0-4c7e-a416-34b5f71e2056
# ╟─f2e36526-aa4e-41bd-88ea-2bf5b172c1c4
# ╠═6b42b8a3-81c8-4dc6-a640-c2b9a78de284
# ╠═20c58a01-f98d-4cf0-9575-e544f605fe1e
# ╠═36b6d708-7a06-4954-8dd7-4d49278057b7
# ╠═9529f9d1-a905-40e3-bc68-428d85d24fcd
# ╠═a2df8f7e-9077-4150-b4dc-322d3b9591db
# ╠═608c630e-6af9-4199-8121-72cd1eea0c6b
# ╠═573de207-995d-46b4-bd4d-dd792b20924c
# ╟─2180ee59-779b-491c-ae8b-b87f2f4eb530
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

---
category: tools
---

The standard way of writing tests Julia is to write a `test/runtests.jl` file. Running `Pkg.test()` will when start a new Julia process that loads your package and runs this file. 

Starting a new process to run your test might make sense for languages like Python, where spinning up a new process is easy. But in Julia, making a new process and loading your package from scratch can take a very long time. Doing this every time you want to test your code (which, if you’re doing some form of test driven development, might be every thirty seconds) is a recipe for exasperation. 

Fortunately, there’s a way around Julia’s long startup times: the `Revise` module. Don’t use the built in `Pkg.test()` function. Just include the test file after activating the test environment and importing Revise. You'll need to do a little setup the first time you try this out, as the environment won't be instantiated. Run

```julia
] activate test
dev .
instantiate
```

From here on out, it you can test with

```julia
] activate test
using Revise
include("test/runtests.jl")
```

You can even wrap the whole thing in the `entr` function, which will re-run the tests whenever you make a change.

```julia
Revise.entr(["test"]; all=true) do
    include("test/runtests.jl")
end
```

Together with stuff like [TidyTest](https://github.com/dhanak/TidyTest.jl), this makes the testing experience in Julia almost as nice as [`pytest-watch`](https://github.com/dhanak/TidyTest.jl) for Python. 

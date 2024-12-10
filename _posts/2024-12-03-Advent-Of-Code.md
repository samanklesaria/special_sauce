I've been playing Advent of Code this year. I started writing solutions in R, but soon realized I was better served by a more general purpose language. Julia took its place. 

[Day 1](https://adventofcode.com/2024/day/1)

Sum the discrepencies between two sorted lists. 

```julia
day1(x) = sum(abs(sort(x) - sort(y)))
```



[Day 2](https://adventofcode.com/2024/day/2)

Find the number of rows that are either increasing or decreasing and for which increments are at least 1 but no more than 3. 

```julia
day2(reports) = sum(map(reports) do r
	δ = r[2:end] .- r[1:end-1]
	Δ = abs.(δ)
	(all(δ .>= 0) | all(δ .<= 0)) & all(Δ .>= 1) & all(Δ .<= 3)
end)
```

For part two, we allow one of the items in each row to be ignored. 

```julia
day2_2(reports) = sum(map(reports) do r
	mapreduce(|, 1:length(r)) do damped
    rm = r[sparsevec([damped], [true], length(r))]
    δ = rm[2:end] .- rm[1:end-1]
		Δ = abs.(δ)
    (all(δ .>= 0) | all(δ .<= 0)) & all(Δ .>= 1) & all(Δ .<= 3)
	end
end)
```



[Day 3](https://adventofcode.com/2024/day/3)

Find every instance of the pattern `mul(A,B)` where `A` and `B` are positive integers and return the sum of `A*B`. 

```julia
day3(s) = sum(prod(parse.(Int, a)) for a in eachmatch(r"mul\((\d+),(\d+)\)", s))
```

For part 2, ignore patterns after the string `don't()` and before the string `do()`. 

```julia
day3_2(s) = foldl(eachmatch(r"do\(\)|don't\(\)|mul\((\d+),(\d+)\)", s), init=0=>true) do (sofar, state), a
	if a.match == "do()"
		sofar=>true
	elseif a.match == "don't()"
		sofar=>false
	elseif state
		(sofar + prod(parse.(Int, a)))=>state
	else
		sofar=>state
	end
end
```



[Day 4](https://adventofcode.com/2024/day/4)

Find the number of times the string "XMAS" can be found in a given word search. 

```julia
function day4(img)
    sum(
        try
            "XMAS" == join([img[i + a * dx, j + a * dy] for a in 0:3])
        catch
            false
        end
        for i in 1:size(img, 1) for j in 1:size(img, 2)
        for dx in [-1, 0, 1] for dy in [-1, 0, 1]
    )
end
```

For part two, find the number of times two "MAS" strings meet in an X shape. 

```julia
function day4_2(img)
    sum(
		any("MAS" == join([img[i + d * a, j + d * a] for a in -1:1])
			for d in (-1,1)) &&
		any("MAS" == join([img[i + d * a, j - d * a] for a in -1:1]) for d in (-1,1))
        for i in 2:(size(img, 1)-1) for j in 2:(size(img, 2) - 1)
    )
end
```



[Day 5](https://adventofcode.com/2024/day/5)

Find the subset of sequences that obey a set of "comes before" rules. Sum their middle elements. For part two, rearrange the the disqualified sequences so that they no longer break the ordering rules and sum the rearrangements' middle elements as well. Here I return the solutions to both parts at once. 

```julia
function day5(pred_rules, seqs)
	h = Set(pred_rules)
	sorted = map(seqs) do s
		sort(s; lt=(a,b)->(a=>b)∈h) == s
	end
  mask = sorted .== seqs
  map([sorted[mask], sorted[.~mask]]) do sub_sorted
    sum([s[ceil(Int, length(s) / 2)] for s in sub_sorted])
  end
end
```



[Day 6](https://adventofcode.com/2024/day/6)

Collect the states visited by a specific automaton. The state is given by an ascii grid of characters. 

```julia
function day6(dims, pos, map)
	dir = [-1, 0]
	encountered = Set([pos])
	while true
		while pos + dir ∉ map
			pos = pos .+ dir
			if any(pos .<= 0) || any(pos .> dims)
				return encountered
			else
				push!(encountered, pos)
			end
		end
		dir = [0 1; -1 0] * dir
	end
end

function parse_day6(grid)
	map = Set(collect.(Tuple.(findall(x->x=='#', grid))))
	pos = collect(Tuple(findfirst(x->x=='^', grid)))
	(shape(grid), pos, map)
end
```



[Day 7](https://adventofcode.com/2024/day/7)

Check whether it's possible to get a given result value by inserting some sequence of `*` and `+`  operators in between a given list of arguments. Sum the result values  for which this is possible. Part 2 adds a digit concatenation operator. 

```julia
function combine(a,b)
	(a * 10^(1 + trunc(Int, log10(b)))) + b
end

ops = FunctionWrapper{Int,Tuple{Int,Int}}[+, *, combine]

function day7(eqs)
	mask = [
		any(result == foldl(zip(chosen, args[2:end]), init=args[1]) do x, (f, y)
			f(x,y)
		end for chosen in Iterators.product(fill(ops, length(args) - 1)...))
	for (args, result) in eqs]
	sum(last.(eqs[mask]))
end
```



[Day 8](https://adventofcode.com/2024/day/8)

... I've realized that abstract descriptions of problems are increasingly going to be impossible to provide. Click the links to read the true problem descriptions. 

```julia
function day8(img)
	s = Set{CartesianIndex{2}}()
	l = DefaultDict{Char, Vector{CartesianIndex{2}}}(Vector{CartesianIndex{2}})
	for ix in CartesianIndices(img)
		if img[ix] != '.'
			push!(l[img[ix]], ix)
		end
	end
	for ixs in values(l)
		for (i,j) in Iterators.product(ixs, ixs)
			if i == j continue end
			dx = j - i
			for a in 1:size(img, 1)
				anode = i + a * dx
				if checkbounds(Bool, img, anode)
					push!(s, anode)
				else
					break
				end
			end
		end
	end
	s
end
```



[Day 9](https://adventofcode.com/2024/day/9)

Part 1:

```julia
function day9(s)
	offset = 0
	spaces = Pair{Int,Int}[]
	files = SortedDict{Int, Pair{Int, Int}}()
	for (i, c) in enumerate(parse.(Int, collect(s)))
		if i % 2 == 0
			if c > 0 push!(spaces, offset=>c) end
		else
			files[offset] = div(i - 1, 2)=>c
		end
		offset += c
	end
	reverse!(spaces)
	while !isempty(spaces) && !isempty(files)
		(old_offset, (id, f_amt)) = poplast!(files)
		(offset, s_amt) = pop!(spaces)
		if old_offset <= offset return files end
		stored_amt = min(f_amt, s_amt)
		files[offset] = id=>stored_amt
		f_resid = f_amt - stored_amt
		if f_resid > 0
			files[old_offset] = id=>f_resid
		end
		s_resid = s_amt - stored_amt
		if s_resid > 0
			push!(spaces, (offset + stored_amt)=>s_resid)
		end
	end
	files
end

checksum(a) = sum(sum(id * (pos:(pos + amt - 1))) for (pos, (id, amt)) in a)
```

Part 2 is much the same, but we use a `SortedDict` for `spaces` instead of a vector. 

```julia
new_files = copy(files)
while !isempty(spaces) && !isempty(files)
  (old_offset, (id, f_amt)) = poplast!(files)
  spc = findfirst(s_amt->s_amt >= f_amt, spaces)
  if isnothing(spc) || old_offset <= first(spc) continue end
  (offset, s_amt) = spc
  delete!(spaces, offset)
  stored_amt = min(f_amt, s_amt)
  delete!(new_files, old_offset)
  new_files[offset] = id=>stored_amt
  s_resid = s_amt - stored_amt
  if s_resid > 0
    spaces[offset + stored_amt] = s_resid
  end
end
```



[Day 10](https://adventofcode.com/2024/day/10)

The code below returns the results for both parts. 

```julia
function advent10(a)
	ids = LinearIndices(a)
	origins = Int[]
	targets = Int[]
	g = SimpleDiGraph(length(a))
	for ix in CartesianIndices(a)
		if a[ix] == 0 push!(origins, ids[ix]) end
		if a[ix] == 9 push!(targets, ids[ix]) end
		for d in [CartesianIndex(1, 0), CartesianIndex(0, 1)]
			for b in [1, -1]
				ix2 = ix + b * d
				if checkbounds(Bool, a, ix2) && a[ix2] - a[ix] == 1
					add_edge!(g, ids[ix], ids[ix2])
				end
			end
		end		
	end
  dists = adjacency_matrix(g)^9
	(sum(dists[origins, targets] .== 9), sum(dists[origins, targets] .> 0))
end
```


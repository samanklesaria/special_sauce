I've been wanting to get better at R (as dplyr and ggplot put pandas and seaborn to shame) so I thought I'd try to use it for Advent of Code this year. Overall, the language feels like a shittier version of Julia. Here's what I have so far.

[Day 1](https://adventofcode.com/2024/day/1)

Sum the discrepencies between two sorted lists. 

```R
day1 <- function(x,y) sum(abs(sort(x) - sort(y)))
```

[Day 2](https://adventofcode.com/2024/day/2)

Find the number of rows that are either increasing or decreasing and for which increments are at least 1 but no more than 3. R's negative indexing behavior works well here. 

```R
day2 <- function(reports) {
  sum(apply(reports, 1, function(a) {
    diffs <- a[-length(a)] - a[-1]
    adiffs <- abs(diffs)
    (all(diffs >= 0) | all(diffs <= 0)) & all(adiffs >= 1) & all(adiffs <= 3)
  }))
}
```

[Day 3](https://adventofcode.com/2024/day/3)

Find every instance of the pattern `mul(A,B)` where `A` and `B` are positive integers and return the sum of `A*B`. 

```R
day3 <- function(s) {
  results <- regmatches(s, gregexec("mul\\((\\d+),(\\d+)\\)", s))[[1]]
  sum(as.numeric(results[2,]) * as.numeric(results[3,]))
}
```

The lack of regular expression literals gets annoying here. This is a little nicer in Julia:

```julia
day3(s) = sum(prod(parse.(Int, a)) for a in eachmatch(r"mul\((\d+),(\d+)\)", s))
```



[Day 4](https://adventofcode.com/2024/day/4)

Find the number of times the string "XMAS" can be found in a given word search. 

```R
day4 <- function(img) {
  a <- 0:3
  c <- 0
  for (i in 0:9) for (j in 1:10) {
      for (dx in -1:1) for (dy in -1:1) {
        ix <- 10 * (i + a * dx) + (j + a * dy)
        if (all(ix > 0) & all(ix <= length(img)))
          c <- c + ("XMAS" == paste(img[ix], collapse=""))
      }
  }
  c
}
```

[Day 5](https://adventofcode.com/2024/day/5)

Find the subset of sequences that obey a set of "comes before" rules. Sum their middle elements. Then rearrange the the disqualified sequences so that they no longer break the ordering rules and sum the rearrangements' middle elements as well. 

```julia
function sort_day5(pred_rules, seqs)
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

... yeah, that was in Julia instead. R doesn't allow you to pick your `<=` operator like Julia does, and I didn't feel like re-writing merge sort. 

I've been wanting to get better at R, so I thought I'd try to use it for Advent of Code this year. Here's what I have so far.

[Day 1](https://adventofcode.com/2024/day/1)

Sum the discrepencies between two sorted lists. 

```R
day1 <- function(x,y) sum(abs(sort(x) - sort(y)))
```

[Day 2](https://adventofcode.com/2024/day/2)

Find the number of rows that are either increasing or decreasing and for which increments are at least 1 but no more than 3. 

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

Find every instance of the pattern `mul(A,B)` where `A` and `B` are positive integers and return the sum of `A*B`. 

```R
day3 <- function(s) {
  results <- regmatches(s, gregexec("mul\\((\\d+),(\\d+)\\)", s))[[1]]
  sum(as.numeric(results[2,]) * as.numeric(results[3,]))
}
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


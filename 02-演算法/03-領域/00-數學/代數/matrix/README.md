# matrix

* https://www.geeksforgeeks.org/determinant-of-a-matrix/
* https://math.stackexchange.com/questions/2269267/determinant-of-large-matrices-theres-gotta-be-a-faster-way
* [What is the best and fastest algorithm to calculate the value of a determinant (Java or C++)?](https://www.quora.com/What-is-the-best-and-fastest-algorithm-to-calculate-the-value-of-a-determinant-Java-or-C++)

this is probably not the most efficient algorithm to calculate determinant but this is simplest way when rows/columns are > 4 because it becomes very complex for other algorithms.

the method is to take LU decomposition
this algorithm is O(n^3).
det M = det LU = det L * det U
where both L and U are triangular, the determinant is a product of the diagonal elements of L and U.

## determinent1.py

```sh
$ python determinent1.py 
mat= [[1, 0, 2, -1], [3, 0, 0, 5], [2, 1, 4, -3], [1, 0, 5, 0]]
det= 30.000000000000004
```


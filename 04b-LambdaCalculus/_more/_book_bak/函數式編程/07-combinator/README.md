# Y-Combinator

* [Fixed-Point Combinators in JavaScript](https://medium.com/@adambene/fixed-point-combinators-in-javascript-c214c15ff2f6) (超讚)
    * [Recursion with Combinators in JavaScript](https://medium.com/@adambene/recursion-with-combinators-injavascript-d797451d054d)
    * [Currying in JavaScript ES6](https://medium.com/@adambene/currying-in-javascript-es6-540d2ad09400)

```js
U = g => g(g) // recursion is strict, must curry
Y = g => g( () => Y(g) ) // we made the recursion 'lazy'
Z = g => v => g(Z(g))(v) // explicit currying makes it 'lazy'
```

## Fixed-Point Combinator

y f = f (y f)

```
y f = f (y f)
    = f (f (y f))
    = f (f (f (y f)))
    = ...
```

With the use of combinators we can define self-referencing anonymous functions avoiding the use of variables.


## Reference

* https://en.wikipedia.org/wiki/Fixed-point_combinator#Y_combinator
* [The Y Combinator (Slight Return)](https://mvanier.livejournal.com/2897.html) (範例為 Scheme)
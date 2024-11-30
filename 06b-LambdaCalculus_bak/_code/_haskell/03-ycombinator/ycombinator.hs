-- from: https://gist.github.com/JustinSDK/9c6c82775f422cc87a73
import Unsafe.Coerce

y :: (a -> a) -> a
y = \f -> (\x -> f (unsafeCoerce x x))(\x -> f (unsafeCoerce x x))

-- Ref: [Y Combinator in Haskell](http://stackoverflow.com/questions/4273413/y-combinator-in-haskell)
-- a fibonacci example
main = putStrLn $ show $ y (\fact -> \n -> if n < 2 then 1 else n * fact (n - 1)) 5
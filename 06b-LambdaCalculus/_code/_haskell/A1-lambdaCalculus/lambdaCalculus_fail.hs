{-# LANGUAGE Rank2Types #-}

module LambdaCalculus where

-- Church Booleans
type ChurchBool = forall a. a -> a -> a

true :: ChurchBool
true x _ = x

false :: ChurchBool
false _ y = y

not' :: ChurchBool -> ChurchBool
not' b = b false true

and' :: ChurchBool -> ChurchBool -> ChurchBool
and' p q = p q p

or' :: ChurchBool -> ChurchBool -> ChurchBool
or' p q = p p q

xor' :: ChurchBool -> ChurchBool -> ChurchBool
xor' p q = p (not' q) q

-- Church Numerals
type ChurchNum = forall a. (a -> a) -> a -> a

zero :: ChurchNum
zero _ x = x

succ' :: ChurchNum -> ChurchNum
succ' n f x = f (n f x)

pred' :: ChurchNum -> ChurchNum
pred' n f x = n (\g h -> h (g f)) (const x) id

-- Numeric Operations
add :: ChurchNum -> ChurchNum -> ChurchNum
add m n f x = m f (n f x)

sub :: ChurchNum -> ChurchNum -> ChurchNum
sub m n = add m (neg n)
  where 
    neg n f x = n (pred' f) x

mult :: ChurchNum -> ChurchNum -> ChurchNum
mult m n f x = m (n f) x

pow :: ChurchNum -> ChurchNum -> ChurchNum
pow x y = y x

-- Conversion Utilities
toInt :: ChurchNum -> Int
toInt n = n (+1) 0

fromInt :: Int -> ChurchNum
fromInt 0 = zero
fromInt n = succ' (fromInt (n - 1))

-- Conversion and Display Helpers
churchToInt :: ChurchNum -> Int
churchToInt = toInt

-- Testing
test :: String -> Bool -> IO ()
test desc condition = 
    putStrLn $ (if condition then "[✓] " else "[✗] ") ++ desc

main :: IO ()
main = do
    test "Successor works" (toInt (succ' zero) == 1)
    test "Addition works" (toInt (add (fromInt 2) (fromInt 3)) == 5)
    test "Multiplication works" (toInt (mult (fromInt 2) (fromInt 3)) == 6)
    test "Power works" (toInt (pow (fromInt 2) (fromInt 3)) == 8)
    
    -- Boolean tests
    test "Not works" (not' true == false)
    test "And works" (and' true true == true)
    test "Or works" (or' false true == true)
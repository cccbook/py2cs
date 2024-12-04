-- Church Numerals
type Church a = (a -> a) -> a -> a

zero :: Church a
zero = \_ x -> x

one :: Church a
one = \f x -> f x

two :: Church a
two = \f x -> f (f x)

three :: Church a
three = \f x -> f (f (f x))

-- Successor function
succ :: Church a -> Church a
succ n = \f x -> f (n f x)

-- Addition of Church numerals
add :: Church a -> Church a -> Church a
add m n = \f x -> m f (n f x)

-- Multiplication of Church numerals
mul :: Church a -> Church a -> Church a
mul m n = \f x -> m (n f) x

-- Church Booleans
type ChurchBool = Church (Church a -> Church a -> a)

true :: ChurchBool
true = \x y -> x

false :: ChurchBool
false = \x y -> y

-- If function for Church Booleans
ifThenElse :: ChurchBool -> a -> a -> a
ifThenElse cond x y = cond x y

-- Apply function to Church numeral
apply :: Church a -> (a -> a) -> a -> a
apply n = n

-- Test examples
exampleAdd :: Int
exampleAdd = apply (add one two) (+1) 0

exampleMul :: Int
exampleMul = apply (mul two three) (+1) 0

exampleBool :: String
exampleBool = ifThenElse true "Yes" "No"

main :: IO ()
main = do
    -- 測試加法
    print $ apply (add one two) (+1) 0  -- 應該輸出 3 (1 + 2)
    
    -- 測試乘法
    print $ apply (mul two three) (+1) 0  -- 應該輸出 6 (2 * 3)
    
    -- 測試布林值條件運算
    print $ ifThenElse true "Yes" "No"  -- 應該輸出 "Yes"
    print $ ifThenElse false "Yes" "No"  -- 應該輸出 "No"

fibs :: [Int]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

main :: IO ()
main = print (take 10 fibs)  -- 輸出 [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

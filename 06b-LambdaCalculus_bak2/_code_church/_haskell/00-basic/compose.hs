compose :: [Int] -> [Int]
compose = map (*2) . filter even

main :: IO ()
main = print (compose [1, 2, 3, 4, 5, 6])  -- 輸出 [4, 8, 12]

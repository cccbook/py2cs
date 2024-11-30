squareList :: [Int] -> [Int]
squareList = map (^2)

main :: IO ()
main = print (squareList [1, 2, 3, 4])  -- 輸出 [1, 4, 9, 16]

quickSort :: (Ord a) => [a] -> [a]
quickSort [] = []
quickSort (x:xs) =
  quickSort [y | y <- xs, y <= x] ++ [x] ++ quickSort [y | y <- xs, y > x]

main :: IO ()
main = print (quickSort [3, 1, 4, 1, 5, 9])  -- 輸出 [1, 1, 3, 4, 5, 9]

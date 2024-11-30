infiniteList :: [Int]
infiniteList = [1..]  -- 無窮數列

takeFirst10 :: [Int]
takeFirst10 = take 10 infiniteList  -- 取前 10 個元素

main :: IO ()
main = print takeFirst10  -- 輸出 [1,2,3,4,5,6,7,8,9,10]

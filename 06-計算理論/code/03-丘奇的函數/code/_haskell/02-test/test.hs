-- 定義 test' 函數
test' :: String -> Bool -> IO ()
test' description assertion = do
  putStrLn description
  putStrLn (if assertion then "[✓]" else "[✗]")

-- 定義 assert 函數（在這裡可以省略）
-- assert :: Bool -> Bool
-- assert condition = condition

-- 階乘函數
factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- 常數 _five
_five :: Int
_five = 5

-- 主函數
main :: IO ()
main = do
  test' "FACTORIAL" (factorial _five == 120)
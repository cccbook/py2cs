data Shape = Circle Float | Rectangle Float Float
  deriving Show

area :: Shape -> Float
area (Circle r) = pi * r^2
area (Rectangle w h) = w * h

main :: IO ()
main = do
  print (area (Circle 5))          -- 圓的面積，輸出 78.53982
  print (area (Rectangle 4 6))     -- 長方形面積，輸出 24.0

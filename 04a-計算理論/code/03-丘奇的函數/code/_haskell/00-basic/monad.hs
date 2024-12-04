main :: IO ()
main = do
  content <- readFile "example.txt"
  putStrLn content

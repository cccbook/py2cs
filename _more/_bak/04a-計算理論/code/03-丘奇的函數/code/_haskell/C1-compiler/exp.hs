module SimpleCompiler where

import Text.ParserCombinators.Parsec
import Control.Monad

-- 定義 AST
data Expr
    = IntLit Int          -- 整數常量
    | Add Expr Expr       -- 加法
    | Mul Expr Expr       -- 乘法
    deriving (Show)

-- 定義虛擬機指令
data Instruction
    = Push Int            -- 壓入常量
    | AddInstr            -- 加法指令
    | MulInstr            -- 乘法指令
    deriving (Show)

-- 解析器
parseExpr :: Parser Expr
parseExpr = parseTerm `chainl1` (addOp <|> mulOp)
  where
    parseTerm = parseInt <|> parens parseExpr
    parseInt = IntLit . read <$> many1 digit
    parens p = between (char '(') (char ')') p
    addOp = char '+' >> return Add
    mulOp = char '*' >> return Mul

-- 編譯器
compile :: Expr -> [Instruction]
compile (IntLit n)   = [Push n]
compile (Add e1 e2)  = compile e1 ++ compile e2 ++ [AddInstr]
compile (Mul e1 e2)  = compile e1 ++ compile e2 ++ [MulInstr]

-- 虛擬機執行
runVM :: [Instruction] -> Int
runVM = go []
  where
    go (x:y:stack) (AddInstr : instrs) = go ((y + x) : stack) instrs
    go (x:y:stack) (MulInstr : instrs) = go ((y * x) : stack) instrs
    go stack       (Push n : instrs)   = go (n : stack) instrs
    go [result]    []                  = result
    go _           _                   = error "Invalid instruction sequence"

-- 主程式
main :: IO ()
main = do
    putStrLn "輸入表達式:"
    input <- getLine
    case parse parseExpr "" input of
        Left err -> print err
        Right ast -> do
            putStrLn $ "AST: " ++ show ast
            let instructions = compile ast
            putStrLn $ "指令: " ++ show instructions
            let result = runVM instructions
            putStrLn $ "結果: " ++ show result

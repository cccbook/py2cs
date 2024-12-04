import (implicit) Prelude
    ( (++),
      otherwise,
      ($),
      Eq((==)),
      Functor(fmap),
      Num((+)),
      Ord((>=)),
      Read,
      Show,
      Bool(..),
      String,
      Int,
      Maybe(..),
      IO,
      fst,
      snd,
      (!!),
      getLine,
      putStrLn,
      read,
      (&&) )

data Ty
    = TyBool
    | TyArr Ty Ty
    | TyNat
    | TyUnit
    | TyPair Ty Ty
    | TyList Ty
  deriving (Show, Eq, Read)

data Term
    = TmTrue
    | TmFalse
    | TmIf Term Term Term
    | TmVar Int
    | TmAbs String Ty Term
    | TmApp Term Term
    | TmZero 
    | TmSucc Term
    | TmPred Term
    | TmIsZero Term
    | TmPlus Term Term
    | TmMult Term Term
    | TmUnit
    | TmPair Term Term
    | TmFst Term
    | TmSnd Term
    | TmNil Ty
    | TmCons Term Term
    | TmIsNil Term
    | TmHead Term
    | TmTail Term
  deriving (Show, Eq, Read)

data Binding = NameBind | VarBind Ty deriving (Show)

type Context = [(String, Binding)]

addBinding :: Context -> String -> Binding -> Context
addBinding ctx x bind = (x, bind) : ctx

getTypeFromContext :: Context -> Int -> Maybe Ty
getTypeFromContext ctx i = 
    case snd (ctx !! i) of
        VarBind ty -> Just ty
        _ -> Nothing

shift :: Int -> Term -> Term
shift d t = f 0 t
  where 
    f c t = case t of 
        TmVar x         -> if x >= c then TmVar (x + d) else TmVar x
        TmAbs x tyT1 t2 -> TmAbs x tyT1 (f (c + 1) t2)
        TmApp t1 t2     -> TmApp (f c t1) (f c t2)
        TmIf t1 t2 t3   -> TmIf (f c t1) (f c t2) (f c t3)
        TmTrue          -> TmTrue
        TmFalse         -> TmFalse
        TmZero          -> TmZero
        TmSucc t1       -> TmSucc (f c t1)
        TmPred t1       -> TmPred (f c t1)
        TmIsZero t1     -> TmIsZero (f c t1)
        TmPlus t1 t2    -> TmPlus (f c t1) (f c t2)
        TmMult t1 t2    -> TmMult (f c t1) (f c t2)
        TmUnit          -> TmUnit
        TmPair t1 t2    -> TmPair (f c t1) (f c t2)
        TmFst t1        -> TmFst (f c t1)
        TmSnd t1        -> TmSnd (f c t1)
        TmNil tyT       -> TmNil tyT
        TmCons t1 t2    -> TmCons (f c t1) (f c t2)
        TmIsNil t1      -> TmIsNil (f c t1)
        TmHead t1       -> TmHead (f c t1)
        TmTail t1       -> TmTail (f c t1)

subst :: Int -> Term -> Term -> Term
subst j s t = f 0 t
  where
    f c t = case t of
        TmVar x         -> if x == j + c then shift c s else TmVar x
        TmAbs x tyT1 t2 -> TmAbs x tyT1 (f (c + 1) t2)
        TmApp t1 t2     -> TmApp (f c t1) (f c t2)
        TmIf t1 t2 t3   -> TmIf (f c t1) (f c t2) (f c t3)
        TmTrue          -> TmTrue
        TmFalse         -> TmFalse
        TmZero          -> TmZero
        TmSucc t        -> TmSucc (f c t)
        TmPred t        -> TmPred (f c t)
        TmIsZero t      -> TmIsZero (f c t)
        TmPlus t1 t2    -> TmPlus (f c t1) (f c t2)
        TmMult t1 t2    -> TmMult (f c t1) (f c t2)
        TmUnit          -> TmUnit
        TmPair t1 t2    -> TmPair (f c t1) (f c t2)
        TmFst t1        -> TmFst (f c t1)
        TmSnd t1        -> TmSnd (f c t1)
        TmNil tyT       -> TmNil tyT
        TmCons t1 t2    -> TmCons (f c t1) (f c t2)
        TmIsNil t1      -> TmIsNil (f c t1)
        TmHead t1       -> TmHead (f c t1)
        TmTail t1       -> TmTail (f c t1)

beta :: Term -> Term -> Term
beta s t = shift (-1) (subst 0 (shift 1 s) t)

isNumerical :: Term -> Bool
isNumerical t =
  case t of 
    TmZero    -> True
    TmSucc t1 -> isNumerical t1
    TmPred t1 -> isNumerical t1
    _         -> False

isVal :: Context -> Term -> Bool
isVal ctx t =
    case t of
        TmAbs _ _ _       -> True
        TmTrue            -> True
        TmFalse           -> True
        TmZero            -> True
        TmUnit            -> True
        TmPair v1 v2      -> isVal ctx v1 && isVal ctx v2
        TmNil _           -> True
        TmCons v1 v2      -> isVal ctx v1 && isVal ctx v2
        _ | isNumerical t -> True
        _                 -> False

eval_step :: Context -> Term -> Maybe Term
eval_step ctx t =
    case t of 
       TmApp (TmAbs _ _ t12) v2 | isVal ctx v2 -> Just (beta v2 t12)

       TmApp v1 t2 | isVal ctx v1 ->
        case eval_step ctx t2 of
            Just t2' -> Just (TmApp v1 t2')
            Nothing -> Nothing

       TmApp t1 t2 ->
        case eval_step ctx t1 of
            Just t1' -> Just (TmApp t1' t2)
            Nothing -> Nothing
       
       TmIf TmTrue t2 _  -> Just t2
       TmIf TmFalse _ t3 -> Just t3

       TmIf t1 t2 t3 -> 
        case eval_step ctx t1 of
            Just t1' -> Just (TmIf t1' t2 t3)
            Nothing -> Nothing
        
       TmSucc t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmSucc t1')
          Nothing -> Nothing
        
       TmPred TmZero -> Just TmZero

       TmPred (TmSucc nv1) | isNumerical nv1 ->  Just nv1

       TmPred t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmPred t1')
          Nothing -> Nothing
       
       TmIsZero TmZero -> Just TmTrue
       
       TmIsZero (TmSucc nv1) | isNumerical nv1 -> Just TmFalse

       TmIsZero t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmIsZero t1')
          Nothing -> Nothing
       
       TmPlus TmZero t1 -> Just t1
       TmPlus t1 TmZero ->  Just t1

       TmPlus (TmSucc t1) t2 ->  Just (TmPlus t1 (TmSucc t2))
       TmPlus t1 t2
            | isVal ctx t1 ->
                case eval_step ctx t2 of
                    Just t2' -> Just (TmPlus t1 t2')
                    Nothing -> Nothing
            | otherwise ->
                case eval_step ctx t1 of
                    Just t1' -> Just (TmPlus t1' t2)
                    Nothing -> Nothing
       
       TmMult TmZero _ -> Just TmZero
       TmMult _ TmZero -> Just TmZero

       TmMult (TmSucc t1) t2 -> Just (TmPlus t2 (TmMult t1 t2))
       TmMult t1 t2
            | isVal ctx t1 ->
                case eval_step ctx t2 of
                    Just t2' -> Just (TmMult t1 t2')
                    Nothing -> Nothing
            | otherwise ->
                case eval_step ctx t1 of
                    Just t1' -> Just (TmMult t1' t2)
                    Nothing -> Nothing
       
       TmPair t1 t2
            | isVal ctx t1 && isVal ctx t2 -> Just (TmPair t1 t2)
            | isVal ctx t1 ->
                case eval_step ctx t2 of
                  Just t2' -> Just (TmPair t1 t2')
                  Nothing -> Nothing
            | otherwise ->
                case eval_step ctx t1 of
                  Just t1' -> Just (TmPair t1' t2)
                  Nothing -> Nothing
        
       TmFst (TmPair v1 _) | isVal ctx v1 -> Just v1

       TmFst t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmFst t1')
          Nothing -> Nothing
      
       TmSnd (TmPair _ v2) | isVal ctx v2 -> Just v2

       TmSnd t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmSnd t1')
          Nothing -> Nothing
       
       TmCons v1 t2 
            | isVal ctx v1 -> 
                case eval_step ctx t2 of
                  Just t2' -> Just (TmCons v1 t2')
                  Nothing -> Nothing

       TmCons t1 t2 -> 
         case eval_step ctx t1 of
         Just t1' -> Just (TmCons t1' t2)
         Nothing -> Nothing

       TmIsNil (TmNil _) -> Just TmTrue

       TmIsNil (TmCons _ _) -> Just TmFalse

       TmIsNil t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmIsNil t1')
          Nothing -> Nothing

       TmHead (TmCons v1 _) | isVal ctx v1 -> Just v1

       TmHead t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmHead t1')
          Nothing -> Nothing

       TmTail (TmCons _ v2) | isVal ctx v2 -> Just v2

       TmTail t1 -> 
        case eval_step ctx t1 of
          Just t1' -> Just (TmTail t1')
          Nothing -> Nothing
      
       _ -> Nothing

typeof :: Context -> Term -> Maybe Ty
typeof ctx t = 
    case t of 
      TmTrue -> Just TyBool
      TmFalse -> Just TyBool

      TmIf t1 t2 t3 -> do
        tyT1 <- typeof ctx t1
        if tyT1 == TyBool
          then do
            tyT2 <- typeof ctx t2
            tyT3 <- typeof ctx t3
            if tyT2 == tyT3 then Just tyT2 else Nothing
        else Nothing

      TmVar i -> getTypeFromContext ctx i 

      TmAbs x tyT1 t2 ->
        let ctx' = addBinding ctx x (VarBind tyT1)
            tyT2 = typeof ctx' t2
         in fmap (TyArr tyT1) tyT2
      
      TmApp t1 t2 -> do
        tyT1 <- typeof ctx t1
        tyT2 <- typeof ctx t2
        case tyT1 of
          TyArr tyT11 tyT12 ->
            if tyT2 == tyT11
              then Just tyT12
              else Nothing
      
      TmZero -> Just TyNat

      TmSucc t1 -> do
        tyT1 <- typeof ctx t1
        if tyT1 == TyNat then Just TyNat else Nothing
      
      TmPred t1 -> do
        tyT1 <- typeof ctx t1
        if tyT1 == TyNat then Just TyNat else Nothing
      
      TmIsZero t1 -> do
        tyT1 <- typeof ctx t1
        if tyT1 == TyNat then Just TyBool else Nothing

      TmPlus t1 t2 -> do
        tyT1 <- typeof ctx t1
        tyT2 <- typeof ctx t2
        if tyT1 == TyNat && tyT2 == TyNat then Just TyNat else Nothing
      
      TmMult t1 t2 -> do
        tyT1 <- typeof ctx t1
        tyT2 <- typeof ctx t2
        if tyT1 == TyNat && tyT2 == TyNat then Just TyNat else Nothing

      TmUnit -> Just TyUnit

      TmPair t1 t2 -> do
        tyT1 <- typeof ctx t1
        tyT2 <- typeof ctx t2
        Just (TyPair tyT1 tyT2)
      
      TmFst t1 -> do
        tyT1 <- typeof ctx t1
        case tyT1 of
          TyPair tyT11 _ -> Just tyT11
          _ -> Nothing
      
      TmSnd t1 -> do
        tyT1 <- typeof ctx t1
        case tyT1 of
          TyPair _ tyT12 -> Just tyT12
          _ -> Nothing
      
      TmNil tyT -> Just (TyList tyT)

      TmCons t1 t2 -> do
        tyT1 <- typeof ctx t1
        tyT2 <- typeof ctx t2
        case tyT2 of
          TyList tyElem ->
            if tyT1 == tyElem then Just tyT2 else Nothing
          _ -> Nothing
      
      TmIsNil t1 -> do
        tyT1 <- typeof ctx t1
        case tyT1 of
          TyList _ -> Just TyBool
          _ -> Nothing

      TmHead t1 -> do
        tyT1 <- typeof ctx t1
        case tyT1 of
          TyList tyElem -> Just tyElem
          _ -> Nothing

      TmTail t1 -> do
        tyT1 <- typeof ctx t1
        case tyT1 of
          TyList _ -> Just tyT1
          _ -> Nothing

evalLoop :: Context -> Term -> Maybe Term
evalLoop ctx t = 
    case eval_step ctx t of
        Just t' -> evalLoop ctx t'
        Nothing -> Just t

eval :: Context -> Term -> Maybe Term
eval ctx t = 
    case typeof ctx t of
        Just _  -> evalLoop ctx t
        Nothing -> Nothing

printTy :: Ty -> String
printTy TyBool           = "Bool"
printTy (TyArr ty1 ty2)  = "(" ++ printTy ty1 ++ " -> " ++ printTy ty2 ++ ")"
printTy TyNat            = "Nat"
printTy TyUnit           = "Unit"
printTy (TyPair ty1 ty2) = "(" ++ printTy ty1 ++ " * " ++ printTy ty2 ++ ")"
printTy (TyList ty)      = "[" ++ printTy ty ++ "]"

printTerm :: Context -> Term -> String
printTerm _ TmTrue            = "true"
printTerm _ TmFalse           = "false"
printTerm _ TmZero            = "0"
printTerm ctx (TmVar x)       = fst (ctx !! x)
printTerm ctx (TmAbs x ty t)  = "(\\lambda " ++ x ++ " : " ++ printTy ty ++ ". " ++ printTerm ((x, NameBind) : ctx) t ++ ")"
printTerm ctx (TmApp t1 t2)   = "(" ++ printTerm ctx t1 ++ " " ++ printTerm ctx t2 ++ ")"
printTerm ctx (TmIf t1 t2 t3) = "if " ++ printTerm ctx t1 ++ " then " ++ printTerm ctx t2 ++ " else " ++ printTerm ctx t3
printTerm ctx (TmSucc t1)     = "succ(" ++ printTerm ctx t1 ++ ")"
printTerm ctx (TmPred t1)     = "pred(" ++ printTerm ctx t1 ++ ")"
printTerm ctx (TmIsZero t1)   = "iszero(" ++ printTerm ctx t1 ++ ")"
printTerm ctx (TmPlus t1 t2)  = "(" ++ printTerm ctx t1 ++ " + " ++ printTerm ctx t2 ++ ")"
printTerm ctx (TmMult t1 t2)  = "(" ++ printTerm ctx t1 ++ " * " ++ printTerm ctx t2 ++ ")"
printTerm ctx (TmPair t1 t2)  = "(" ++ printTerm ctx t1 ++ ", " ++ printTerm ctx t2 ++ ")"
printTerm ctx (TmFst t)       = "fst(" ++ printTerm ctx t ++ ")"
printTerm ctx (TmSnd t)       = "snd(" ++ printTerm ctx t ++ ")"
printTerm _ TmUnit            = "unit"
printTerm _ (TmNil ty)        = "nil[" ++ printTy ty ++ "]"
printTerm ctx (TmCons t1 t2)  = "cons(" ++ printTerm ctx t1 ++ ", " ++ printTerm ctx t2 ++ ")"
printTerm ctx (TmIsNil t1)    = "isnil(" ++ printTerm ctx t1 ++ ")"
printTerm ctx (TmHead t1)     = "head(" ++ printTerm ctx t1 ++ ")"
printTerm ctx (TmTail t1)     = "tail(" ++ printTerm ctx t1 ++ ")"

-- Read eval print loop
repl :: Context -> IO ()
repl ctx = do
    putStrLn "Enter an expression: "
    l <- getLine
    let expression = read l :: Term
    case typeof ctx expression of
        Just ty -> do
            putStrLn $ "Type: " ++ printTy ty
            case eval ctx expression of
                Just result -> putStrLn $ "Result: " ++ printTerm ctx result
                Nothing -> putStrLn "Evaluation error."
        Nothing -> putStrLn "Type error."
    repl ctx

main :: IO ()
main = repl []
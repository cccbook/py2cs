(ns lambda-calculus.core)

;; Church Boolean implementation
(defn church-bool [b]
  (fn [x y] (if b x y)))

(def TRUE (church-bool true))
(def FALSE (church-bool false))

(defn NOT [b]
  (fn [x y] ((b y x))))

(defn AND [p]
  (fn [q]
    ((p q) FALSE)))

(defn OR [p]
  (fn [q]
    ((p TRUE) q)))

(defn XOR [p]
  (fn [q]
    ((p (NOT q)) q)))

(defn ASSERT [truth]
  (fn [description]
    (if (truth true false)
      (str "[✓] " description)
      (str "[✗] " description))))

(defn REFUTE [truth]
  (ASSERT (NOT truth)))

(defn TEST [description assertion]
  (println (assertion description)))

(defn run-tests []
  (TEST "TRUE"
    (ASSERT TRUE))
  
  (TEST "FALSE"
    (REFUTE FALSE))
  
  (TEST "AND"
    (ASSERT (((AND TRUE) TRUE))))
  
  (TEST "OR"
    (ASSERT 
      ((AND 
         (((OR TRUE) FALSE))
         (((OR FALSE) TRUE)))
       (NOT (((OR FALSE) FALSE))))))
  
  (TEST "XOR"
    (ASSERT 
      ((AND 
         (((XOR TRUE) FALSE))
         (((XOR FALSE) TRUE)))
       (NOT (((XOR TRUE) TRUE))))))
  
  (TEST "NOT"
    (REFUTE (NOT TRUE))))

;; Run the tests
(run-tests)
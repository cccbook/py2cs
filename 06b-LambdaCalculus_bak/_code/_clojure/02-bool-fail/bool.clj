(ns lambda-calculus.core)

;; Lambda Calculus data structures using closures

;; Church Booleans : Logic
(def IF (fn [c] (fn [x] (fn [y] ((c x) y)))))

(def TRUE (fn [x] (fn [y] x)))

(def FALSE (fn [x] (fn [y] y)))

(def AND (fn [p] (fn [q] ((p q) p))))

(def OR (fn [p] (fn [q] ((p p) q))))

(def NOT (fn [c] ((c FALSE) TRUE)))

(def XOR (fn [p] (fn [q] ((p (NOT q)) q))))

(defn ASSERT [truth]
  (fn [description] 
    (if ((truth TRUE) FALSE)
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
    (ASSERT ((AND TRUE) TRUE)))
  
  (TEST "OR"
    (ASSERT 
      ((AND 
         ((OR TRUE) FALSE))
         ((OR FALSE) TRUE))
       (NOT ((OR FALSE) FALSE)))))
  
  (TEST "XOR"
    (ASSERT 
      ((AND 
         ((XOR TRUE) FALSE)
         ((XOR FALSE) TRUE))
       (NOT ((XOR TRUE) TRUE)))))
  
  (TEST "NOT"
    (REFUTE (NOT TRUE))))

;; Run the tests
(run-tests)
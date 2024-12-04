;; from -- https://gist.github.com/rahulr92/4fc389fe67e63db13a65a1d20c4297f9
(ns rahul.lambda-calculus)

;; Church Numerals
(def zero (fn [f]
            (fn [x]
              x)))

(def one (fn [f]
           (fn [x]
             (f x))))

(def two (fn [f]
           (fn [x]
             (f (f x)))))

(def to-int  (fn [n]
               ((n (fn [i]
                     (+ i 1)))
                0)))

;; (to-int two) => 2

;; Successor function
(def succ (fn [n]
            (fn [f]
              (fn [x]
                (f ((n f) x))))))

;; (to-int (succ two)) => 3
;; (to-int (succ (succ two))) => 4

;; Arithmetic

;; Addition
(def add (fn [n]
           (fn [m]
             (fn [f]
               (fn [x]
                 ((m f) ((n f) x)))))))

;; (to-int ((add one) two)) => 3

;; Multiplication
(def mul (fn [n]
           (fn [m]
             (fn [f]
               (fn [x]
                 ((m (n f)) x))))))

;; (to-int ((mul two) two)) => 4
;; (to-int ((mul two) one)) => 2

;; n to the power of m
(def power (fn [n]
             (fn [m]
               (fn [f]
                 (fn [x]
                   (((m n) f) x))))))

;; (def three ((add two) one))
;; (to-int ((power two) three)) => 8

;; shorter version of power
(def power2 (fn [n]
                   (fn [m]
                     (m n))))

;; (to-int ((power2 two) three)) => 8

;; Conditionals
(def troo (fn [then-do]
            (fn [else-do]
              then-do)))

(def falz (fn [then-do]
            (fn [else-do]
              else-do)))

(def ifthenelse (fn [conditn]
                  (fn [then-do]
                    (fn [else-do]
                      ((conditn then-do) else-do)))))

;;  (def tired troo)

;;  (def coffees_today
;;    (((ifthenelse tired)
;;      three)
;;     one))

;; (to-int coffees_today) => 3

;; Logic
(def opposite (fn [boolean]
                  (fn [then-do]
                    (fn [else-do]
                      ((boolean else-do) then-do)))))

(def to-bool (fn [boolean]
               ((boolean true) false)))

;; (to-bool troo) => true
;; (to-bool falz) => false

;; (to-bool (opposite troo)) => false
;; (to-bool (opposite falz)) => true

;; Predicates
(def is-zero (fn [n]
               ((n (fn [_]
                     falz))
                troo)))

;; (to-bool (is-zero zero)) => true
;; (to-bool (is-zero one)) => false

(def is-even (fn [n]
               ((n opposite) troo)))

;; (to-bool (is-even one)) => false
;; (to-bool (is-even two)) => true

(def both (fn [boola]
            (fn [boolb]
              ((boola boolb)
               boola))))
;; (to-bool ((both troo) troo)) => true
;; (to-bool ((both troo) falz)) => false
;; (to-bool ((both falz) falz)) => false
;; (to-bool ((both falz) troo)) => false

(def inc-or (fn [boola]
              (fn [boolb]
                ((boola boola)
                 boolb))))
;; (to-bool ((inc-or troo) troo)) => true
;; (to-bool ((inc-or troo) falz)) => true
;; (to-bool ((inc-or falz) falz)) => false
;; (to-bool ((inc-or falz) troo)) => true

;; Data structures

(def make-pair (fn [left]
                 (fn [right]
                   (fn [f]
                     ((f left) right)))))

(def left (fn [pair]
            (pair troo)))

(def right (fn [pair]
             (pair falz)))

;; (to-int
;;  (right
;;   ((make-pair three) two))) => 2

;; the empty list
(def null ((make-pair troo) troo))

(def is-empty left)

;; (to-bool (is-empty null)) => true

(def prepend (fn [item]
               (fn [l]
                 ((make-pair falz)
                  ((make-pair item) l)))))

(def head (fn [l]
            (left (right l))))

(def tail (fn [l]
            (right (right l))))

;; (to-int (head
;;  ((prepend two) null))) => 2

;; (to-int (head
;;  (tail
;;   ((prepend three)
;;     ((prepend one)
;;      ((prepend two) null)))))) => 1

(def three ((add two) one))

(ns rahul.lambda-calculus-test
  (:require [clojure.test :refer :all]
            [rahul.lambda-calculus :refer :all]))

;; Helper function to test Church Numerals
(defn church-numeral-test [numeral expected-int]
  (is (= expected-int (to-int numeral))))

(deftest church-numerals-test
  (testing "Church Numeral Conversions"
    (church-numeral-test zero 0)
    (church-numeral-test one 1)
    (church-numeral-test two 2)))

(deftest successor-function-test
  (testing "Successor function"
    (church-numeral-test (succ zero) 1)
    (church-numeral-test (succ two) 3)
    (church-numeral-test (succ (succ two)) 4)))

(deftest arithmetic-test
  (testing "Addition"
    (church-numeral-test ((add zero) one) 1)
    (church-numeral-test ((add one) two) 3)
    (church-numeral-test ((add two) two) 4))
  
  (testing "Multiplication"
    (church-numeral-test ((mul zero) one) 0)
    (church-numeral-test ((mul two) two) 4)
    (church-numeral-test ((mul two) one) 2))
  
  (testing "Power"
    (church-numeral-test ((power two) zero) 1)
    (church-numeral-test ((power two) one) 2)
    (church-numeral-test ((power two) two) 4)
    (church-numeral-test ((power two) three) 8)
    
    (church-numeral-test ((power2 two) zero) 1)
    (church-numeral-test ((power2 two) one) 2)
    (church-numeral-test ((power2 two) two) 4)
    (church-numeral-test ((power2 two) three) 8)))

(deftest boolean-logic-test
  (testing "Boolean Conversion"
    (is (= true (to-bool troo)))
    (is (= false (to-bool falz))))
  
  (testing "Opposite/Negation"
    (is (= false (to-bool (opposite troo))))
    (is (= true (to-bool (opposite falz)))))
  
  (testing "Conditional (if-then-else)"
    (church-numeral-test 
     (((ifthenelse troo) three) one) 3)
    (church-numeral-test 
     (((ifthenelse falz) three) one) 1)))

(deftest predicate-test
  (testing "Is Zero Predicate"
    (is (= true (to-bool (is-zero zero))))
    (is (= false (to-bool (is-zero one))))
    (is (= false (to-bool (is-zero two)))))
  
  (testing "Is Even Predicate"
    (is (= false (to-bool (is-even one))))
    (is (= true (to-bool (is-even two))))
    (is (= false (to-bool (is-even three))))
    (is (= true (to-bool (is-even ((mul two) two))))))
  
  (testing "Logical Combinators"
    (testing "Both"
      (is (= true (to-bool ((both troo) troo))))
      (is (= false (to-bool ((both troo) falz))))
      (is (= false (to-bool ((both falz) falz))))
      (is (= false (to-bool ((both falz) troo)))))
    
    (testing "Inclusive OR"
      (is (= true (to-bool ((inc-or troo) troo))))
      (is (= true (to-bool ((inc-or troo) falz))))
      (is (= false (to-bool ((inc-or falz) falz))))
      (is (= true (to-bool ((inc-or falz) troo)))))))

(deftest pair-and-list-test
  (testing "Pair Operations"
    (let [test-pair ((make-pair three) two)]
      (church-numeral-test (left test-pair) 3)
      (church-numeral-test (right test-pair) 2)))
  
  (testing "List Operations"
    (is (= true (to-bool (is-empty null))))
    
    (let [test-list ((prepend two) null)
          test-list-multi ((prepend three)
                           ((prepend one)
                            ((prepend two) null)))]
      (church-numeral-test (head test-list) 2)
      (church-numeral-test (head (tail test-list-multi)) 1))))

(ns rahul.lambda-calculus-test
  (:require [clojure.test :refer :all]
            [rahul.lambda-calculus :refer :all]))

;; [Previous test definitions remain the same]

(defn run-tests []
  (clojure.test/run-tests *ns*))

(run-tests)

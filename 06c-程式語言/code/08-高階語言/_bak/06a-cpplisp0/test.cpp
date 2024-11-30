#include "lisp.cpp"

unsigned g_test_count;      // count of number of unit tests executed
unsigned g_fault_count;     // count of number of unit tests that fail

template <typename T1, typename T2>
void test_equal_(const T1 & value, const T2 & expected_value, const char * file, int line)
{
    ++g_test_count;
    if (value != expected_value) {
        std::cout
            << file << '(' << line << ") : "
            << " expected " << expected_value
            << ", got " << value
            << '\n';
        ++g_fault_count;
    }
}

// write a message to std::cout if value != expected_value
#define TEST_EQUAL(value, expected_value) test_equal_(value, expected_value, __FILE__, __LINE__)

// evaluate the given Lisp expression and compare the result against the given expected_result
#define TEST(expr, expected_result) TEST_EQUAL(to_string(eval(read(expr), &global_env)), expected_result)


int main ()
{
    environment global_env; add_globals(global_env);

    // the 29 unit tests for lis.py
    TEST("(quote (testing 1 (2.0) -3.14e159))", "(testing 1 (2.0) -3.14e159)");
    TEST("(+ 2 2)", "4");
    TEST("(+ (* 2 100) (* 1 10))", "210");
    TEST("(if (> 6 5) (+ 1 1) (+ 2 2))", "2");
    TEST("(if (< 6 5) (+ 1 1) (+ 2 2))", "4");
    TEST("(define x 3)", "3");
    TEST("x", "3");
    TEST("(+ x x)", "6");
    TEST("(begin (define x 1) (set! x (+ x 1)) (+ x 1))", "3");
    TEST("((lambda (x) (+ x x)) 5)", "10");
    TEST("(define twice (lambda (x) (* 2 x)))", "<Lambda>");
    TEST("(twice 5)", "10");
    TEST("(define compose (lambda (f g) (lambda (x) (f (g x)))))", "<Lambda>");
    TEST("((compose list twice) 5)", "(10)");
    TEST("(define repeat (lambda (f) (compose f f)))", "<Lambda>");
    TEST("((repeat twice) 5)", "20");
    TEST("((repeat (repeat twice)) 5)", "80");
    TEST("(define fact (lambda (n) (if (<= n 1) 1 (* n (fact (- n 1))))))", "<Lambda>");
    TEST("(fact 3)", "6");
    //TEST("(fact 50)", "30414093201713378043612608166064768844377641568960512000000000000");
    TEST("(fact 12)", "479001600"); // no bignums; this is as far as we go with 32 bits
    TEST("(define abs (lambda (n) ((if (> n 0) + -) 0 n)))", "<Lambda>");
    TEST("(list (abs -3) (abs 0) (abs 3))", "(3 0 3)");
    TEST("(define combine (lambda (f)"
             "(lambda (x y)"
                "(if (null? x) (quote ())"
                "(f (list (car x) (car y))"
                "((combine f) (cdr x) (cdr y)))))))", "<Lambda>");
    TEST("(define zip (combine cons))", "<Lambda>");
    TEST("(zip (list 1 2 3 4) (list 5 6 7 8))", "((1 5) (2 6) (3 7) (4 8))");
    TEST("(define riff-shuffle (lambda (deck) (begin"
            "(define take (lambda (n seq) (if (<= n 0) (quote ()) (cons (car seq) (take (- n 1) (cdr seq))))))"
            "(define drop (lambda (n seq) (if (<= n 0) seq (drop (- n 1) (cdr seq)))))"
            "(define mid (lambda (seq) (/ (length seq) 2)))"
            "((combine append) (take (mid deck) deck) (drop (mid deck) deck)))))", "<Lambda>");
    TEST("(riff-shuffle (list 1 2 3 4 5 6 7 8))", "(1 5 2 6 3 7 4 8)");
    TEST("((repeat riff-shuffle) (list 1 2 3 4 5 6 7 8))",  "(1 3 5 7 2 4 6 8)");
    TEST("(riff-shuffle (riff-shuffle (riff-shuffle (list 1 2 3 4 5 6 7 8))))", "(1 2 3 4 5 6 7 8)");

    std::cout
        << "total tests " << g_test_count
        << ", total failures " << g_fault_count
        << "\n";
    return g_fault_count ? EXIT_FAILURE : EXIT_SUCCESS;
}

# 密碼學

## 因數分解

* [維基百科 : 整數分解](https://zh.wikipedia.org/wiki/%E6%95%B4%E6%95%B0%E5%88%86%E8%A7%A3)
    * [試除法](https://zh.wikipedia.org/wiki/%E8%AF%95%E9%99%A4%E6%B3%95)
    * [埃拉托斯特尼篩法](https://zh.wikipedia.org/wiki/%E5%9F%83%E6%8B%89%E6%89%98%E6%96%AF%E7%89%B9%E5%B0%BC%E7%AD%9B%E6%B3%95)
    * [米勒-拉賓質數判定法](https://zh.wikipedia.org/wiki/%E7%B1%B3%E5%8B%92-%E6%8B%89%E5%AE%BE%E6%A3%80%E9%AA%8C)
    * [費馬質數判定法](https://zh.wikipedia.org/wiki/%E8%B4%B9%E9%A9%AC%E7%B4%A0%E6%80%A7%E6%A3%80%E9%AA%8C)
    * [維基百科 : 秀爾演算法](https://zh.wikipedia.org/wiki/%E7%A7%80%E7%88%BE%E6%BC%94%E7%AE%97%E6%B3%95) (量子算法！)

## 理論基礎

* [維基百科:有限體](https://zh.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E5%9F%9F)
    * https://en.wikipedia.org/wiki/Finite_field

## 加解密

* [Node.js加密算法库Crypto](http://blog.fens.me/nodejs-crypto/) (讚！)
* https://nodejs.org/api/crypto.html
* [維基百科:密碼雜湊函式](https://zh.wikipedia.org/wiki/%E5%AF%86%E7%A2%BC%E9%9B%9C%E6%B9%8A%E5%87%BD%E6%95%B8)
    * https://en.wikipedia.org/wiki/Cryptographic_hash_function
* [RSA and ECC in JavaScript](http://www-cs-students.stanford.edu/~tjw/jsbn/)
    * 包含 SHA1
* [Which symmetric key algorithm does SSL use?](https://stackoverflow.com/questions/6088583/which-symmetric-key-algorithm-does-ssl-use)
    * Symmetric algorithms supported in SSL are DES, 3DES, ARCFOUR, AES, Camellia, RC2, IDEA, SEED, NULL (no encryption).
* [維基百科:RSA加密演算法](https://zh.wikipedia.org/zh-tw/RSA%E5%8A%A0%E5%AF%86%E6%BC%94%E7%AE%97%E6%B3%95)
    * http://www.mathland.idv.tw/life/rsa576.htm
* [維基百科:橢圓曲線密碼學](https://zh.wikipedia.org/wiki/%E6%A4%AD%E5%9C%86%E6%9B%B2%E7%BA%BF%E5%AF%86%E7%A0%81%E5%AD%A6)
    * 橢圓曲線密碼學（英語：Elliptic curve cryptography，縮寫為 ECC）
* [List of random number generators](https://en.wikipedia.org/wiki/List_of_random_number_generators)
    * https://en.wikipedia.org/wiki/Blum_Blum_Shub (x[n+1] = x[n]^2 mod M)
    * https://en.wikipedia.org/wiki/Linear_congruential_generator (x[n+1] = a x[n] + c mod M)
    * https://en.wikipedia.org/wiki/Lehmer_random_number_generator (x[n+1] = a x[n] mod M)
    * https://en.wikipedia.org/wiki/Yao%27s_test (偽隨機數通過此測試者，難以和真的隨機數做區分)

## 區塊鏈

* [比特幣 (Bit Coin) 的運作原理](http://pansci.asia/archives/53571) -- 陳鍾誠
* [A blockchain in 200 lines of code](https://medium.com/@lhartikk/a-blockchain-in-200-lines-of-code-963cc1cc0e54)
    * https://github.com/lhartikk/naivechain
    * https://lhartikk.github.io/
* [Blockchain Demo](https://anders.com/blockchain/)
* https://anders.com/blockchain/
    * https://github.com/anders94/blockchain-demo/

* https://en.wikipedia.org/wiki/List_of_important_publications_in_cryptography

## Number 數論

* http://numbers.github.io/index.html
  * [numbers.js, 面向 node.js 和JavaScript的高級數學庫](http://hant.helplib.com/GitHub/article_93511)

## 大整數

* https://github.com/MikeMcl/decimal.js
* https://github.com/MikeMcl/bignumber.js
* https://github.com/MikeMcl/big.js

## 質數測試

* https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
* https://github.com/indutny/miller-rabin/blob/master/lib/mr.js
* https://github.com/indutny/primal

## 大數相乘

* [Karatsuba算法](https://zh.wikipedia.org/wiki/Karatsuba%E7%AE%97%E6%B3%95)
* https://gist.github.com/haocong/c2d9b2169d28eb15a94d
* https://stackoverflow.com/questions/28372569/implementing-karatsuba-multiplication-in-javascript

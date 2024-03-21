% https://github.com/abdulmlik/simple-Prolog-Examples/blob/master/fact.pl
% Author: Abdulmalik Ben Ali
% Date: 5/15/2017

fact(1,1):-!.
fact(N,R):- integer(N),N1 is N -1 , fact(N1,R1) , R is R1 *N.
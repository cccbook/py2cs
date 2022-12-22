module Nand(input a, b, output out);
  nand g1(out, a, b);
endmodule

module And(input a, b, output out);
  Nand g1(a, b, AnandB);
  Nand g2(AnandB, AnandB, out);
endmodule


module main;
reg a, b;
wire abAnd;
And  g3(a, b, abAnd);
initial
begin
  $monitor("%4dns b=%d a=%d abAnd=%d", 
            $stime, b, a,  abAnd);
  a  = 0;
  b  = 0;
end

always #50 begin
  a = a+1;
end

always #100 begin
  b = b+1;
end

initial #500 $finish;
endmodule

export class Complex {
  a:number
  b:number

  constructor(a:number,b:number) { this.a = a; this.b = b; }
  
  conj() { return new Complex(this.a, -1*this.b); }
  
  add(c2:Complex) { return new Complex(this.a+c2.a, this.b+c2.b); }
  
  sub(c2:Complex) { return new Complex(this.a-c2.a, this.b-c2.b); }
  
  mul(c2:Complex) {
    var a=this.a, b=this.b, c=c2.a, d=c2.b;
    return new Complex(a*c-b*d, a*d+b*c);
  }
  
  div(c2:Complex) {
    var a=this.a, b=this.b, c=c2.a, d=c2.b;
    return new Complex((a*c+b*d)/(c*c+d*d), (b*c-a*d)/(c*c+d*d));
  }
  
  toString() { return this.a+'+'+this.b+'i'; }
  /*
  parse(s:string) {
    try {
      var m = s.match(/^([^\+]*)(\+(.*))?$/);
      var a = parseFloat(m[1]);
      var b = typeof m[3]==='undefined'?1:parseFloat(m[3]);
      return new Complex(a, b)  
    } catch (e) {
      throw new Error(`Complex:parse($s) error!`)
    }
  }
  */
  
  ln() {
    var a=this.a, b=this.b, r=a*a+b*b;
    var w = 1/2*Math.log(r);
    var x = Math.acos(a/Math.sqrt(r));
    return new Complex(w, x);
  }
  
  exp() {
    var a=this.a, b=this.b;
    var r=Math.exp(a);
    return new Complex(r*Math.cos(b), r*Math.sin(b));
  }
}

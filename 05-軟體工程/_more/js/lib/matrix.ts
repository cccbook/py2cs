import * as U from './util.ts'
import * as V from './vector.ts'

export function clone(m:number[][]) {
  return JSON.parse(JSON.stringify(m))
}

export function matrix(rows:number, cols:number) {
  let r = new Array(rows)
  for (let i=0; i<rows; i++) {
    r[i] = V.vector(cols, 0)
  }
  return r
}

export function flatten(m:number[][]) {
  let rows = m.length, cols = m[0].length
  let r = new Array()
  for (let i=0; i<rows; i++) {
    for (let j=0; j<cols; j++)
    r[i*cols+j] = m[i][j]
  }
  return r
}

export function identity(n:number) {
  let v = V.vector(n, 1)
  return diag(v)
}

export function diag(v:number[]) {
  let rows = v.length
  let r = matrix(rows, rows)
  for (let i = 0; i < rows; i++) {
    r[i][i] = v[i]
  }
  return r
}

export function transpose(m:number[][]) {
  let r = []
  let rows = m.length
  let cols = m[0].length
  for (let j = 0; j < cols; j++) {
    let rj:number[] = r[j] = []
    for (let i = 0; i < rows; i++) {
      rj[i] = m[i][j]
    }
  }
  return r
}

export let tr = transpose

export function dot(a:number[][], b:number[][]) {
  let arows = a.length
  let bcols = b[0].length
  let r = []
  let bt = tr(b)
  for (let i = 0; i < arows; i++) {
    let ri:number[] = r[i] = []
    for (let j = 0; j < bcols; j++) {
      ri.push(V.dot(a[i], bt[j]))
    }
  }
  return r
}

export function inv(m0:number[][]) {
  let m = m0.length, n = m0[0].length, abs = Math.abs
  let A = clone(m0), Ai, Aj
  let I = identity(m), Ii, Ij
  let i, j, k, x, i0, v0
  for (j = 0; j < n; ++j) {
    i0 = -1
    v0 = -1
    for (i = j; i !== m; ++i) {
      k = abs(A[i][j])
      if (k > v0) { i0 = i; v0 = k }
    }
    Aj = A[i0]; A[i0] = A[j]; A[j] = Aj
    Ij = I[i0]; I[i0] = I[j]; I[j] = Ij
    x = Aj[j]
    for (k = j; k !== n; ++k) Aj[k] /= x
    for (k = n - 1; k !== -1; --k) Ij[k] /= x
    for (i = m - 1; i !== -1; --i) {
      if (i !== j) {
        Ai = A[i]
        Ii = I[i]
        x = Ai[j]
        for (k = j + 1; k !== n; ++k) Ai[k] -= Aj[k] * x
        for (k = n - 1; k > 0; --k) { Ii[k] -= Ij[k] * x; --k; Ii[k] -= Ij[k] * x }
        if (k === 0) Ii[0] -= Ij[0] * x
      }
    }
  }
  return I
}

export function det(x:number[][]) {
  let abs = Math.abs
  if (x.length !== x[0].length) { throw new Error('numeric: det() only works on square matrices') }
  let n = x.length, ret = 1, i, j, k, A = clone(x), Aj, Ai, alpha, temp, k1
  for (j = 0; j < n - 1; j++) {
    k = j
    for (i = j + 1; i < n; i++) { if (abs(A[i][j]) > abs(A[k][j])) { k = i } }
    if (k !== j) {
      temp = A[k]; A[k] = A[j]; A[j] = temp
      ret *= -1
    }
    Aj = A[j]
    for (i = j + 1; i < n; i++) {
      Ai = A[i]
      alpha = Ai[j] / Aj[j]
      for (k = j + 1; k < n - 1; k += 2) {
        k1 = k + 1
        Ai[k] -= Aj[k] * alpha
        Ai[k1] -= Aj[k1] * alpha
      }
      if (k !== n) { Ai[k] -= Aj[k] * alpha }
    }
    if (Aj[j] === 0) { return 0 }
    ret *= Aj[j]
  }
  return ret * A[j][j]
}

// AX = b
export function lu(A:number[][]) {
  var abs = Math.abs
  var i, j, k, absAjk, Akk, Ak, Pk, Ai
  var max
  var n = A.length, n1 = n-1
  var P = new Array(n)
  A = clone(A)

  for (k = 0; k < n; ++k) {
    Pk = k
    Ak = A[k]
    max = abs(Ak[k])
    for (j = k + 1; j < n; ++j) {
      absAjk = abs(A[j][k])
      if (max < absAjk) {
        max = absAjk
        Pk = j
      }
    }
    P[k] = Pk

    if (Pk != k) {
      A[k] = A[Pk]
      A[Pk] = Ak
      Ak = A[k]
    }

    Akk = Ak[k]

    for (i = k + 1; i < n; ++i) {
      A[i][k] /= Akk
    }

    for (i = k + 1; i < n; ++i) {
      Ai = A[i]
      for (j = k + 1; j < n1; ++j) {
        Ai[j] -= Ai[k] * Ak[j]
        ++j
        Ai[j] -= Ai[k] * Ak[j]
      }
      if(j===n1) Ai[j] -= Ai[k] * Ak[j]
    }
  }

  return { LU: A, P:  P }
}

export function luSolve(LUP:any, b:number[]) {
  var i, j;
  var LU = LUP.LU;
  var n   = LU.length;
  var x:number[] = U.clone(b);
  var P   = LUP.P;
  var Pi, LUi, LUii, tmp;

  for (i=n-1;i!==-1;--i) x[i] = b[i];
  for (i = 0; i < n; ++i) {
    Pi = P[i];
    if (P[i] !== i) {
      tmp = x[i];
      x[i] = x[Pi];
      x[Pi] = tmp;
    }

    LUi = LU[i];
    for (j = 0; j < i; ++j) {
      x[i] -= x[j] * LUi[j];
    }
  }

  for (i = n - 1; i >= 0; --i) {
    LUi = LU[i];
    for (j = i + 1; j < n; ++j) {
      x[i] -= x[j] * LUi[j];
    }

    x[i] /= LUi[i];
  }

  return x;
}

export function solve(A:number[][],b:number[]) { return luSolve(lu(A), b) }

// ============================ SVD =======================================================
const Epsilon = 2.220446049250313e-16

export function svd(A:number[][]) {
  var temp;
//Compute the thin SVD from G. H. Golub and C. Reinsch, Numer. Math. 14, 403-420 (1970)
var prec= Epsilon; //Math.pow(2,-52) // assumes double prec
var tolerance= 1.e-64/prec;
var itmax= 50;
var c=0;
var i=0;
var j=0;
var k=0;
var l=0;

var u= clone(A);
var m= u.length;

var n= u[0].length;

if (m < n) throw "Need more rows than columns"

var e = new Array(n);
var q = new Array(n);
for (i=0; i<n; i++) e[i] = q[i] = 0.0;
var v = matrix(n, n); 
//	v.zero();

function pythag(a:number,b:number)
 {
  a = Math.abs(a)
  b = Math.abs(b)
  if (a > b)
    return a*Math.sqrt(1.0+(b*b/a/a))
  else if (b == 0.0) 
    return a
  return b*Math.sqrt(1.0+(a*a/b/b))
}

//Householder's reduction to bidiagonal form

var f= 0.0;
var g= 0.0;
var h= 0.0;
var x= 0.0;
var y= 0.0;
var z= 0.0;
var s= 0.0;

for (i=0; i < n; i++)
{	
  e[i]= g;
  s= 0.0;
  l= i+1;
  for (j=i; j < m; j++) 
    s += (u[j][i]*u[j][i]);
  if (s <= tolerance)
    g= 0.0;
  else
  {	
    f= u[i][i];
    g= Math.sqrt(s);
    if (f >= 0.0) g= -g;
    h= f*g-s
    u[i][i]=f-g;
    for (j=l; j < n; j++)
    {
      s= 0.0
      for (k=i; k < m; k++) 
        s += u[k][i]*u[k][j]
      f= s/h
      for (k=i; k < m; k++) 
        u[k][j]+=f*u[k][i]
    }
  }
  q[i]= g
  s= 0.0
  for (j=l; j < n; j++) 
    s= s + u[i][j]*u[i][j]
  if (s <= tolerance)
    g= 0.0
  else
  {	
    f= u[i][i+1]
    g= Math.sqrt(s)
    if (f >= 0.0) g= -g
    h= f*g - s
    u[i][i+1] = f-g;
    for (j=l; j < n; j++) e[j]= u[i][j]/h
    for (j=l; j < m; j++)
    {	
      s=0.0
      for (k=l; k < n; k++) 
        s += (u[j][k]*u[i][k])
      for (k=l; k < n; k++) 
        u[j][k]+=s*e[k]
    }	
  }
  y= Math.abs(q[i])+Math.abs(e[i])
  if (y>x) 
    x=y
}

// accumulation of right hand gtransformations
for (i=n-1; i != -1; i+= -1)
{	
  if (g != 0.0)
  {
     h= g*u[i][i+1]
    for (j=l; j < n; j++) 
      v[j][i]=u[i][j]/h
    for (j=l; j < n; j++)
    {	
      s=0.0
      for (k=l; k < n; k++) 
        s += u[i][k]*v[k][j]
      for (k=l; k < n; k++) 
        v[k][j]+=(s*v[k][i])
    }	
  }
  for (j=l; j < n; j++)
  {
    v[i][j] = 0;
    v[j][i] = 0;
  }
  v[i][i] = 1;
  g= e[i]
  l= i
}

// accumulation of left hand transformations
for (i=n-1; i != -1; i+= -1)
{	
  l= i+1
  g= q[i]
  for (j=l; j < n; j++) 
    u[i][j] = 0;
  if (g != 0.0)
  {
    h= u[i][i]*g
    for (j=l; j < n; j++)
    {
      s=0.0
      for (k=l; k < m; k++) s += u[k][i]*u[k][j];
      f= s/h
      for (k=i; k < m; k++) u[k][j]+=f*u[k][i];
    }
    for (j=i; j < m; j++) u[j][i] = u[j][i]/g;
  }
  else
    for (j=i; j < m; j++) u[j][i] = 0;
  u[i][i] += 1;
}

// diagonalization of the bidiagonal form
prec= prec*x
for (k=n-1; k != -1; k+= -1)
{
  for (var iteration=0; iteration < itmax; iteration++)
  {	// test f splitting
    var test_convergence = false
    for (l=k; l != -1; l+= -1)
    {	
      if (Math.abs(e[l]) <= prec)
      {	test_convergence= true
        break 
      }
      if (Math.abs(q[l-1]) <= prec)
        break 
    }
    if (!test_convergence)
    {	// cancellation of e[l] if l>0
      c= 0.0
      s= 1.0
      var l1= l-1
      for (i =l; i<k+1; i++)
      {	
        f= s*e[i]
        e[i]= c*e[i]
        if (Math.abs(f) <= prec)
          break
        g= q[i]
        h= pythag(f,g)
        q[i]= h
        c= g/h
        s= -f/h
        for (j=0; j < m; j++)
        {	
          y= u[j][l1]
          z= u[j][i]
          u[j][l1] =  y*c+(z*s)
          u[j][i] = -y*s+(z*c)
        } 
      }	
    }
    // test f convergence
    z= q[k]
    if (l== k)
    {	//convergence
      if (z<0.0)
      {	//q[k] is made non-negative
        q[k]= -z
        for (j=0; j < n; j++)
          v[j][k] = -v[j][k]
      }
      break  //break out of iteration loop and move on to next k value
    }
    if (iteration >= itmax-1)
      throw 'Error: no convergence.'
    // shift from bottom 2x2 minor
    x= q[l]
    y= q[k-1]
    g= e[k-1]
    h= e[k]
    f= ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
    g= pythag(f,1.0)
    if (f < 0.0)
      f= ((x-z)*(x+z)+h*(y/(f-g)-h))/x
    else
      f= ((x-z)*(x+z)+h*(y/(f+g)-h))/x
    // next QR transformation
    c= 1.0
    s= 1.0
    for (i=l+1; i< k+1; i++)
    {	
      g= e[i]
      y= q[i]
      h= s*g
      g= c*g
      z= pythag(f,h)
      e[i-1]= z
      c= f/z
      s= h/z
      f= x*c+g*s
      g= -x*s+g*c
      h= y*s
      y= y*c
      for (j=0; j < n; j++)
      {	
        x= v[j][i-1]
        z= v[j][i]
        v[j][i-1] = x*c+z*s
        v[j][i] = -x*s+z*c
      }
      z= pythag(f,h)
      q[i-1]= z
      c= f/z
      s= h/z
      f= c*g+s*y
      x= -s*g+c*y
      for (j=0; j < m; j++)
      {
        y= u[j][i-1]
        z= u[j][i]
        u[j][i-1] = y*c+z*s
        u[j][i] = -y*s+z*c
      }
    }
    e[l]= 0.0
    e[k]= f
    q[k]= x
  } 
}
  
//vt= transpose(v)
//return (u,q,vt)
for (i=0;i<q.length; i++) 
  if (q[i] < prec) q[i] = 0
  
//sort eigenvalues	
for (i=0; i< n; i++)
{	 
//writeln(q)
 for (j=i-1; j >= 0; j--)
 {
  if (q[j] < q[i])
  {
//  writeln(i,'-',j)
   c = q[j]
   q[j] = q[i]
   q[i] = c
   for(k=0;k<u.length;k++) { temp = u[k][i]; u[k][i] = u[k][j]; u[k][j] = temp; }
   for(k=0;k<v.length;k++) { temp = v[k][i]; v[k][i] = v[k][j]; v[k][j] = temp; }
//	   u.swapCols(i,j)
//	   v.swapCols(i,j)
   i = j	   
  }
 }	
}

return {U:u,S:q,V:v}
};


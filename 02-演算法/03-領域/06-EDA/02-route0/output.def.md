

```
VERSION 5.8 ;
DESIGN top ;
UNITS DISTANCE MICRONS 1000 ;
DIEAREA ( 0 0 ) ( 100 100 ) ;
COMPONENTS 15 ;
- gate__00_ AND
  + PLACED ( 12 12 ) N ;
  + INPUTS ( a[3] b[3] ) ;
- gate__01_ XOR
  + PLACED ( 37 12 ) N ;
  + INPUTS ( a[3] b[3] ) ;
- gate__02_ AND
  + PLACED ( 62 12 ) N ;
  + INPUTS ( a[2] b[2] ) ;
- gate__03_ AND
  + PLACED ( 87 12 ) N ;
  + INPUTS ( _01_ _02_ ) ;
- gate__04_ OR
  + PLACED ( 12 37 ) N ;
  + INPUTS ( _03_ _00_ ) ;
- gate__05_ XOR
  + PLACED ( 37 37 ) N ;
  + INPUTS ( a[2] b[2] ) ;
- gate__06_ AND
  + PLACED ( 62 37 ) N ;
  + INPUTS ( _01_ _05_ ) ;
- gate__07_ AND
  + PLACED ( 87 37 ) N ;
  + INPUTS ( a[1] b[1] ) ;
- gate__08_ XOR
  + PLACED ( 12 62 ) N ;
  + INPUTS ( a[1] b[1] ) ;
- gate__09_ AND
  + PLACED ( 37 62 ) N ;
  + INPUTS ( a[0] b[0] ) ;
- gate__10_ AND
  + PLACED ( 62 62 ) N ;
  + INPUTS ( _09_ _08_ ) ;
- gate__11_ AND
  + PLACED ( 87 62 ) N ;
  + INPUTS ( _07_ _10_ ) ;
- gate__12_ AND
  + PLACED ( 12 87 ) N ;
  + INPUTS ( _06_ _11_ ) ;
- gate__13_ OR
  + PLACED ( 37 87 ) N ;
  + INPUTS ( _11_ _05_ ) ;
- gate__14_ OR
  + PLACED ( 62 87 ) N ;
  + INPUTS ( _13_ _02_ ) ;
END COMPONENTS
NETS 16 ;
- _00_
  + ROUTED ( 12 12 ) ( 12 12 ) ( 12 37 ) ;
- _01_
  + ROUTED ( 37 12 ) ( 87 12 ) ( 87 12 ) ;
- _01_
  + ROUTED ( 37 12 ) ( 62 12 ) ( 62 37 ) ;
- _02_
  + ROUTED ( 62 12 ) ( 87 12 ) ( 87 12 ) ;
- _02_
  + ROUTED ( 62 12 ) ( 62 12 ) ( 62 87 ) ;
- _03_
  + ROUTED ( 87 12 ) ( 12 12 ) ( 12 37 ) ;
- _05_
  + ROUTED ( 37 37 ) ( 62 37 ) ( 62 37 ) ;
- _05_
  + ROUTED ( 37 37 ) ( 37 37 ) ( 37 87 ) ;
- _06_
  + ROUTED ( 62 37 ) ( 12 37 ) ( 12 87 ) ;
- _07_
  + ROUTED ( 87 37 ) ( 87 37 ) ( 87 62 ) ;
- _08_
  + ROUTED ( 12 62 ) ( 62 62 ) ( 62 62 ) ;
- _09_
  + ROUTED ( 37 62 ) ( 62 62 ) ( 62 62 ) ;
- _10_
  + ROUTED ( 62 62 ) ( 87 62 ) ( 87 62 ) ;
- _11_
  + ROUTED ( 87 62 ) ( 12 62 ) ( 12 87 ) ;
- _11_
  + ROUTED ( 87 62 ) ( 37 62 ) ( 37 87 ) ;
- _13_
  + ROUTED ( 37 87 ) ( 62 87 ) ( 62 87 ) ;
END NETS
END DESIGN
```

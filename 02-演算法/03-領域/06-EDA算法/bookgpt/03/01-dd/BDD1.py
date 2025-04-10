from dd.autoref import BDD

bdd = BDD()
bdd.declare('x', 'y', 'z', 'w')

# conjunction (in TLA+ syntax)
u = bdd.add_expr(r'x /\ y')
    # symbols `&`, `|` are supported too
    # note the "r" before the quote,
    # which signifies a raw string and is
    # needed to allow for the backslash
print(u.support)
# substitute variables for variables (rename)
rename = dict(x='z', y='w')
v = bdd.let(rename, u)
# substitute constants for variables (cofactor)
values = dict(x=True, y=False)
v = bdd.let(values, u)
# substitute BDDs for variables (compose)
d = dict(x=bdd.add_expr(r'z \/ w'))
v = bdd.let(d, u)
# as Python operators
v = bdd.var('z') & bdd.var('w')
v = ~ v
# quantify universally ("forall")
u = bdd.add_expr(r'\A x, y:  (x /\ y) => y')
# quantify existentially ("exist")
u = bdd.add_expr(r'\E x, y:  x \/ y')
# less readable but faster alternative,
# (faster because of not calling the parser;
# this may matter only inside innermost loops)
u = bdd.var('x') | bdd.var('y')
u = bdd.exist(['x', 'y'], u)
assert u == bdd.true, u
# inline BDD references
u = bdd.add_expr(rf'x /\ {v}')
# satisfying assignments (models):
# an assignment
d = bdd.pick(u, care_vars=['x', 'y'])
# iterate over all assignments
for d in bdd.pick_iter(u):
    print(d)
# how many assignments
n = bdd.count(u)
# write to and load from JSON file
filename = 'bdd.json'
bdd.dump(filename, roots=dict(res=u))
other_bdd = BDD()
roots = other_bdd.load(filename)
print(other_bdd.vars)
class Gate:
    count = 0
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.id = Gate.count
        Gate.count += 1
    def normalForm(self):
        plist = []
        for param in self.params:
            if isinstance(param, Gate):
                plist.append(param.normalForm())
            else:
                plist.append("_")
        return f"{self.name}({','.join(plist)})"
    def allNodes(self):
        nodes = []
        allNodes(self, nodes)
        return nodes

def allNodes(gate, nodes):
    nodes.append(gate)
    for param in gate.params:
        if isinstance(param, Gate):
            allNodes(param, nodes)

def Not(a):
    return Gate("not", [a])

def Nand(a,b):
    return Gate("nand", [a,b])

class GateLib:
    def __init__(self, gateLib):
        self.gateLib = gateLib
        self.gmap = {}
        for name, array in gateLib.items():
            area = array[0]; gates=array[1:]
            for g in gates:
                self.gmap[g.normalForm()]={'name':name, 'area':area, 'gate':g }

    def dump(self):
        for name, array in gateLib.items():
            area = array[0]; gates=array[1:]
            print(name, ' ', area, end =" ")
            for g in gates:
                print(g.normalForm(), end=" ")
            print()

    def find(self, g):
        gexp = g.normalForm() if isinstance(g, Gate) else g
        return self.gmap.get(gexp)

a = "_"; b= "_"; c="_"; d="_"; e="_"; f="_"; g="_"; h="_"

gateLib = {
"NOT"    :[2, Not(a)],
"NAND"   :[3, Nand(a,b)],
"NAND3"  :[4, Nand(Not(Nand(a,b)),c)],
"NAND4"  :[5, Nand(Not(Nand(Not(Nand(a,b)),c)),d),
              Nand(Not(Nand(a,b)),Not(Nand(c,d)))],
"AOI21"  :[4, Not(Nand(Nand(a,b),Not(c)))],
"AOI22"  :[5, Not(Nand(Nand(a,b),Nand(c,d)))]
}

if __name__ == '__main__':
    glib = GateLib(gateLib)
    glib.dump()
    print(glib.find('not(_)'))


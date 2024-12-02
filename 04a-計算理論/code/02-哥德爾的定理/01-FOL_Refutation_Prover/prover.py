import random
import re
import itertools
import time

OPS = ["IF", "AND", "OR", "NOT", "IMPLIES"]
QUANTS = ["FORALL", "EXISTS"]


def getElement(xtype, xvalue):
    return (xtype, xvalue)


def remove_conditionals(FOL_Tree):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()

    if _symbol_type == "op" and _symbol_value == "IMPLIES":
        _symbol_value = "OR"
        FOL_Tree.set_node(_symbol_type, _symbol_value)

        _child_node = _child_nodes[-1]

        # _child_symbol_type=child_node.get_element_type()
        # _child_symbol_value=child_node.get_element_value()

        new_symbol_type = "op"
        new_symbol_value = "NOT"
        _new_node = Node(new_symbol_type, new_symbol_value)

        _new_node.add_child(_child_node)

        _child_nodes[-1] = _new_node

        FOL_Tree.set_child_nodes(_child_nodes)

    _child_nodes = FOL_Tree.get_child_nodes()
    for i in range(len(_child_nodes)):
        remove_conditionals(_child_nodes[i])

    return FOL_Tree


def deMorgan(FOL_Tree):
    _current_node = FOL_Tree
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()

    if _symbol_type == "op" and _symbol_value == "NOT":

        _child_node = _child_nodes[0]

        _child_symbol_type = _child_node.get_element_type()
        _child_symbol_value = _child_node.get_element_value()

        # _symbol_value="OR"
        # FOL_Tree.set_node(_symbol_type,_symbol_value)

        new_symbol_type = new_symbol_value = ""
        if _child_symbol_type == "op":
            new_symbol_type = "op"
            if _child_symbol_value == "AND":
                new_symbol_value = "OR"
            elif _child_symbol_value == "OR":
                new_symbol_value = "AND"
        elif _child_symbol_type == "quant":
            new_symbol_type = "quant"
            if _child_symbol_value == "FORALL":
                new_symbol_value = "EXISTS"
            elif _child_symbol_value == "EXISTS":
                new_symbol_value = "FORALL"

        if new_symbol_type != "" and new_symbol_value != "":
            FOL_Tree.set_node(new_symbol_type, new_symbol_value)

            _child_child_nodes = _child_node.get_child_nodes()

            _new_children = []
            for _child_child_node in _child_child_nodes:

                _child_child_symbol_type = _child_child_node.get_element_type()
                _child_child_symbol_value = _child_child_node.get_element_value()

                if _child_child_symbol_type == "variable":

                    _new_children.append(_child_child_node)

                else:

                    _new_node = Node(_symbol_type, _symbol_value)
                    _new_node.add_child(_child_child_node)

                    _new_children.append(_new_node)

            FOL_Tree.set_child_nodes(_new_children)

    _child_nodes = FOL_Tree.get_child_nodes()
    for i in range(len(_child_nodes)):
        deMorgan(_child_nodes[i])

    return FOL_Tree


def standardize(FOL_Tree, variable_names={}):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()

    if _symbol_type == "quant":
        _child_node = _child_nodes[-1]

        _child_symbol_type = _child_node.get_element_type()
        _child_symbol_value = _child_node.get_element_value()

        variable_names[_child_symbol_value] = _child_symbol_value + "_" + str(random.randint(0, 10000))

        _child_node.set_node(_child_symbol_type, variable_names[_child_symbol_value])

        _child_nodes[-1] = _child_node

    elif _symbol_type == "function" or _symbol_type == "predicate":
        for i in range(len(_child_nodes)):

            _child_node = _child_nodes[i]
            _child_symbol_type = _child_node.get_element_type()
            _child_symbol_value = _child_node.get_element_value()

            if _child_symbol_value in variable_names:
                _child_node.set_node(_child_symbol_type, variable_names[_child_symbol_value])

            _child_nodes[i] = _child_node

    FOL_Tree.set_child_nodes(_child_nodes)

    for i in range(len(_child_nodes)):
        standardize(_child_nodes[i], variable_names)

    return FOL_Tree


def recorrect(FOL_Tree):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()

    for i in range(len(_child_nodes)):
        _child_node=_child_nodes[i]

        _child_node_symbol_type = _child_node.get_element_type()
        _child_node_symbol_value = _child_node.get_element_value()

        if (_symbol_type == "function"):
            if (_child_node_symbol_type == "function"):
                _symbol_type = "predicate"
                FOL_Tree.set_node(_symbol_type, _symbol_value)

        if (_symbol_type == "op" or _symbol_type == "quant"):
            if (_child_node_symbol_type == "function"):
                _child_node_symbol_type = "predicate"
                _child_node.set_node(_child_node_symbol_type,_child_node_symbol_value)

                _child_nodes[i]=_child_node

        if (_symbol_type == "predicate"):
            if (_child_node_symbol_type == "symbol"):
                _child_node_symbol_type = "variable"
                _child_node.set_node(_child_node_symbol_type, _child_node_symbol_value)

                _child_nodes[i] = _child_node

        recorrect(_child_node)

    FOL_Tree.set_child_nodes(_child_nodes)

    return FOL_Tree

def isCNF(FOL_Tree):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()

    if (_symbol_type == "op" and _symbol_value == "AND"):

        for i in range(len(_child_nodes)):
            _child_node=_child_nodes[i]

            _child_node_symbol_type = _child_node.get_element_type()
            _child_node_symbol_value = _child_node.get_element_value()

            if not (_child_node_symbol_type == "op" and _child_node_symbol_value == "OR"):
                return False

            _child_child_nodes =_child_node.get_child_nodes()

            for i in range(len(_child_child_nodes)):
                _child_child_node = _child_child_nodes[i]

                _child_child_node_symbol_type = _child_child_node.get_element_type()
                _child_child_node_symbol_value = _child_child_node.get_element_value()

                if (_child_child_node_symbol_type == "op" and (_child_child_node_symbol_value == "AND" or _child_child_node_symbol_value == "OR")):
                    return False



    else:
        return False

    return True

def isClause(FOL_Tree):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()

    if (_symbol_type == "op" and _symbol_value == "OR"):
        for i in range(len(_child_nodes)):
            _child_node = _child_nodes[i]

            _child_node_symbol_type = _child_node.get_element_type()
            _child_node_symbol_value = _child_node.get_element_value()

            if _child_node_symbol_type == "op" and _child_node_symbol_value == "NOT":
                continue;

            elif _child_node_symbol_type == "predicate":
                continue;

            else:
                return False

    else:
        return False

    return True

def isLiteral(FOL_Tree):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()

    if (_symbol_type == "op" and _symbol_value == "NOT"):
        _child_node = _child_nodes[0]

        _child_node_symbol_type = _child_node.get_element_type()
        _child_node_symbol_value = _child_node.get_element_value()

        if _child_node_symbol_type == "predicate":
            return True

    if _symbol_type == "predicate":
        return True

    return False



def concatenate(node_tuple):
    _new_children=[]
    for node in node_tuple:
        _new_children.extend(node.get_child_nodes())

    parent=Node("op","OR")
    parent.set_child_nodes(_new_children)
    return parent


def convertToCNF(FOL_Tree):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    _child_nodes = FOL_Tree.get_child_nodes()
    _no_child_nodes=len(_child_nodes)

    ## CNF check
    if isCNF(FOL_Tree):
        return FOL_Tree

    ## clause
    if isClause(FOL_Tree):
        _new_FOL_Tree = Node("op", "AND")
        _new_FOL_Tree.set_child_nodes([FOL_Tree])

        return _new_FOL_Tree


    ## Literal
    if isLiteral(FOL_Tree):
        _new_FOL_Tree = Node("op", "AND")

        _new_parent_node=Node("op", "OR")
        _new_parent_node.set_child_nodes([FOL_Tree])

        _new_FOL_Tree.set_child_nodes([_new_parent_node])

        return _new_FOL_Tree



    if _symbol_type == "op" and _symbol_value == "AND" and _no_child_nodes>0:
        _new_children=[]
        for i in range(len(_child_nodes)):
            _child_node =_child_nodes[i]
            X_Tree = convertToCNF(_child_node)
            x_child_nodes = X_Tree.get_child_nodes()
            for x_child_node in x_child_nodes:
                _new_children.append(x_child_node)

        _new_FOL_Tree=Node("op", "AND")
        _new_FOL_Tree.set_child_nodes(_new_children)

        return _new_FOL_Tree

    if _symbol_type == "op" and _symbol_value == "OR" and _no_child_nodes > 0:
        _new_children = []
        for i in range(len(_child_nodes)):
            _child_node = _child_nodes[i]


            X_Tree = convertToCNF(_child_node)

            x_child_nodes=X_Tree.get_child_nodes()
            _x_children=[]
            for x_child_node in x_child_nodes:
                _x_children.append(x_child_node)

            _new_children.append(_x_children)

        _new_combined_children=list(itertools.product(*_new_children))

        _new_x_children=[]
        for node_tuple in _new_combined_children:
            _new_x_node=concatenate(node_tuple)
            _new_x_children.append(_new_x_node)


        _new_FOL_Tree = Node("op", "AND")
        _new_FOL_Tree.set_child_nodes(_new_x_children)

        return _new_FOL_Tree

    else:
        print("Error.")



class Node:
    def __init__(self, element_type, element_value):
        self.set_node(element_type, element_value)

        self.children = []

    def add_child(self, x_node):
        self.children.append(x_node)

    def set_child_nodes(self, children):
        self.children = children

    def get_child_nodes(self):
        return self.children

    def set_node(self, element_type, element_value):
        self.element_type = element_type
        self.element_value = element_value

    def get_element_type(self):
        return self.element_type

    def get_element_value(self):
        return self.element_value

    def get_text(self):
        return "[" + self.get_element_type() + "] " + self.get_element_value()

    def __str__(self, level=0):
        text = "--" * level + self.get_text() + "\n"
        for child_node in self.children:
            text += child_node.__str__(level + 1)
        return text


def printTree(node):
    node.get_child_nodes()


def parse_tree(args):
    _stack = []

    _stack_element = None
    for index in range(len(args)):
        current_index = index

        current_element = args[current_index]
        current_symbol_type = current_element[0]
        current_symbol_value = current_element[1]

        if current_symbol_type == "open_bracket" and current_symbol_value == "(":
            continue
        elif current_symbol_type == "close_bracket" and current_symbol_value == ")":

            _picked_child_nodes = []
            while True:
                _parent_node = _stack.pop()

                _symbol_type = _parent_node.get_element_type()
                _symbol_value = _parent_node.get_element_value()

                if _symbol_type == "op" or _symbol_type == "quant" or _symbol_type == "function" or _symbol_type == "predicate":
                    _child_nodes = _parent_node.get_child_nodes()
                    _no_child_nodes = len(_child_nodes)

                    if _no_child_nodes == 0:
                        break

                _picked_child_nodes.append(_parent_node)

            _parent_node.set_child_nodes(_picked_child_nodes)

            _stack.append(_parent_node)

        else:
            _node = Node(current_symbol_type, current_symbol_value)
            _stack.append(_node)

    assert (len(_stack) == 1)

    return _stack.pop()


def parse(F):
    characters = F
    # breaking the input to argument types
    # argument types: open and close bracket, operator and symbol
    args = []

    regex = r'''\(|\)|\[|\]|\-?\d+\.\d+|\-?\d+|[^,(^)\s]+'''

    # sanitizing the input
    characters = characters.replace("\t", " ")
    characters = characters.replace("\n", " ")
    characters = characters.replace("\r", " ")
    characters = characters.lstrip(" ")
    characters = characters.rstrip(" ")

    ##prev_arg_name = None

    prev_arg = next_arg = None
    lines = []
    arg_list = re.findall(regex, characters)
    for i in range(len(arg_list)):
        arg = arg_list[i]
        if (i - 1 >= 0):
            prev_arg = arg_list[i - 1]
        if (i + 1 < len(arg_list)):
            next_arg = arg_list[i + 1]

        if (arg == "("):
            arg_name = "open_bracket"
        elif (arg == ")"):
            arg_name = "close_bracket"
        elif prev_arg == "(":
            if (arg in OPS):
                arg_name = "op"
            elif (arg in QUANTS):
                arg_name = "quant"
            else:
                arg_name = "function"
        elif (prev_arg in QUANTS):
            arg_name = "variable"
        elif arg.isalnum():
            arg_name = "symbol"

        arg_tuple = (arg_name, arg)
        args.append(arg_tuple)

    return parse_tree(args)

##################################################################################

def findIncSet(F):
    result = []
    for i in range(0, len(F)):
        try:
            F[i] = algorithm(F[i])
            if F[i]:
                result.append(i)
        except BaseException:
            continue

    return result

def algorithm(L):
    clauses = list()
    for i in range(0, len(L)):
        FOL_Tree = parse(L[i])
        FOL_Tree = recorrect(FOL_Tree)
        FOL_Tree = remove_conditionals(FOL_Tree)
        FOL_Tree = deMorgan(FOL_Tree)
        FOL_Tree = doubleNOT(FOL_Tree)
        FOL_Tree = standardize(FOL_Tree)
        FOL_Tree = prenex_form(FOL_Tree)
        FOL_Tree = skolemize(FOL_Tree)
        FOL_Tree = drop_universal(FOL_Tree)
        FOL_Tree = symbol_fixer(FOL_Tree)
        FOL_Tree = convertToCNF(FOL_Tree)

        predicates = and_to_clausal(FOL_Tree)
        clauses.append(predicates)

    return resolution(clauses)


def doubleNOT(FOL_Tree):
    currentNode = FOL_Tree
    _symbol_value = currentNode.get_element_value()
    if len(currentNode.get_child_nodes()) == 0:
        return FOL_Tree

    if _symbol_value == "NOT": #if current is NOT
        child = currentNode.get_child_nodes()
        child_value = child[0].get_element_value()
        if child_value == "NOT": #if child is also not, set current to its child
            currentNode.set_node(currentNode.get_child_nodes()[0].get_child_nodes()[0].get_element_type(), currentNode.get_child_nodes()[0].get_child_nodes()[0].get_element_value())

            if len(currentNode.get_child_nodes()[0].get_child_nodes()[0].get_child_nodes()) == 2:
                currentNode.get_child_nodes()[1] = currentNode.get_child_nodes()[0].get_child_nodes()[0].get_child_nodes()[1]

            currentNode.get_child_nodes()[0] = currentNode.get_child_nodes()[0].get_child_nodes()[0].get_child_nodes()[0]
            
            doubleNOT(currentNode)

        
    if len(currentNode.get_child_nodes()) == 2:
        doubleNOT(currentNode.get_child_nodes()[1])

    doubleNOT(currentNode.get_child_nodes()[0])

    return FOL_Tree

############### PRENEX CODE START ###################
# This function checks if its already in prenex form, if not it passes it to a converter.
def prenex_form(FOL_Tree):
    # Prenex form is pushing all the quantifiers up as far as possible.
    if_prenex = prenex_check(FOL_Tree)

    if if_prenex:
        return FOL_Tree
    else: 
        FOL_Tree = prenex_convert(FOL_Tree)
        return prenex_form(FOL_Tree)

def prenex_check(FOL_Tree):
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    currChildren = FOL_Tree.get_child_nodes()

    # Check for lenghth of children here
    if len(currChildren) != 2:
        return True

    leftChild = currChildren[1]
    leftChildType = leftChild.get_element_type()

    rightChild = currChildren[0]
    rightChildType = rightChild.get_element_type()

    if _symbol_type != "quant" and (leftChildType == "quant" or rightChildType == "quant"):
        return False
    
    else:
        return prenex_check(leftChild) and prenex_check(rightChild)

def prenex_convert(FOL_Tree):
    currentNode = FOL_Tree
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()

    currChildren = FOL_Tree.get_child_nodes()

    if len(currChildren) != 2:
        return FOL_Tree

    leftChild = currChildren[1]
    leftChildType = leftChild.get_element_type()
    leftChildValue = leftChild.get_element_value()
    leftChildChildren = leftChild.get_child_nodes()

    rightChild = currChildren[0]
    rightChildType = rightChild.get_element_type()
    rightChildValue = rightChild.get_element_value()
    rightChildChildren = rightChild.get_child_nodes()

    if _symbol_type == "op" and leftChildType == "quant": #Left subtree is the issue.
        leftChildLeft = leftChildChildren[1]
        leftChildLeftType = leftChildLeft.get_element_type()
        leftChildLeftValue = leftChildLeft.get_element_value()

        tempType = currentNode.get_element_type()
        tempValue = currentNode.get_element_value()

        currentNode.set_node(leftChildType, leftChildValue)
        leftChild.set_node(tempType, tempValue)

        temp = Node(leftChildLeftType, leftChildLeftValue)
        leftChild.get_child_nodes()[1] = leftChild.get_child_nodes()[0]
        leftChild.get_child_nodes()[0] = currentNode.get_child_nodes()[0]
        currentNode.get_child_nodes()[0] = currentNode.get_child_nodes()[1]
        currentNode.get_child_nodes()[1] = temp


    elif _symbol_type == "op" and rightChildType == "quant": #Right subtree is the issue.
        rightChildLeft = rightChildChildren[1]
        rightChildLeftType = rightChildLeft.get_element_type()
        rightChildLeftValue = rightChildLeft.get_element_value()

        tempType = currentNode.get_element_type()
        tempValue = currentNode.get_element_value()

        currentNode.set_node(rightChildType, rightChildValue)
        rightChild.set_node(tempType, tempValue)

        temp = Node(rightChildLeftType, rightChildLeftValue)
        rightChild.get_child_nodes()[1] = currentNode.get_child_nodes()[1]
        currentNode.get_child_nodes()[1] = temp

    prenex_convert(leftChild)
    prenex_convert(rightChild)
    return FOL_Tree

################ PRENEX CODE END ##################

varList = []        
def skolemize(FOL_Tree):
    global varList
    currentNode = FOL_Tree
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()
    
    if _symbol_type != "quant":
        varList = []
        return FOL_Tree
    else:
        currChildren = currentNode.get_child_nodes()

        leftChild = currChildren[1]
        leftChildValue = leftChild.get_element_value()

        rightChild = currChildren[0]
        rightChildType = rightChild.get_element_type()
        rightChildValue = rightChild.get_element_value()

        rightChildChildren = rightChild.get_child_nodes()
        if len(rightChildChildren) == 2:
            rightChildleft = rightChildChildren[1]
        rightChildRight = rightChildChildren[0]
        if _symbol_value == "FORALL":
            varList.append(leftChildValue) 

        else: #EXISTS
            if varList == []:
                new_symbol_type = "symbol"
            else:
                new_symbol_type = "function"
            rename(FOL_Tree, new_symbol_type, leftChildValue)

            currentNode.set_node(rightChildType, rightChildValue)
            if len(rightChildChildren) == 1:
                currChildren.pop(1)

            for i in range(0, len(rightChildChildren)):
                currChildren[i] = rightChildChildren[i]
            skolemize(currentNode)
            
        skolemize(currChildren[0])    
        return FOL_Tree
    
    
def rename(FOL_Tree, new_symbol_type, var):
    global varList
    currentNode = FOL_Tree
    currChildren = currentNode.get_child_nodes()
    _symbol_type = FOL_Tree.get_element_type()
    _symbol_value = FOL_Tree.get_element_value()
    
    if _symbol_value == var:
        if new_symbol_type == "symbol":
            currentNode.set_node(new_symbol_type, var)
        else: #new_symbol_type == "function"
            currentNode.set_node(new_symbol_type, var)
            for i in range(0, len(varList)):
                temp = Node("variable", varList[i])
                currentNode.add_child(temp)

    else:
        for i in range(len(currChildren)):
            FOL_Tree = rename(currChildren[i], new_symbol_type, var)

    return FOL_Tree

universal_varList = []
def drop_universal(FOL_Tree):
    global universal_varList
    _symbol_type = FOL_Tree.get_element_type()
    currentNode = FOL_Tree
    currChildren = currentNode.get_child_nodes()

    if len(currChildren) != 2:
        return FOL_Tree
    else:
        if _symbol_type == "quant":
            leftChild = currChildren[1]
            leftChildValue = leftChild.get_element_value()
            universal_varList.append(leftChildValue)

            rightChild = currChildren[0]
            rightChildType = rightChild.get_element_type()
            rightChildValue = rightChild.get_element_value()

            rightChildChildren = rightChild.get_child_nodes()

            currentNode.set_node(rightChildType, rightChildValue)
            if len(rightChildChildren) == 1:
                currChildren.pop(1)

            for i in range(0, len(rightChildChildren)):
                currChildren[i] = rightChildChildren[i]

            drop_universal(currentNode)
        
        drop_universal(currChildren[0])
        return FOL_Tree

def symbol_fixer(FOL_Tree):
    currentNode = FOL_Tree
    if currentNode.get_element_type() == "variable":
        if currentNode.get_element_value() not in universal_varList:
            currentNode.set_node("symbol", currentNode.get_element_value())

    if len(currentNode.get_child_nodes()) == 0:
        return FOL_Tree

    for i in range(0, len(currentNode.get_child_nodes())):
        symbol_fixer(currentNode.get_child_nodes()[i])
        
    return FOL_Tree
    
###################################################################################

VARIABLE = "VARIABLE"
CONSTANT = "CONSTANT"


class Argument(object):
    def __init__(self, name, kind):
        self._name = name
        self._kind = kind

    def is_variable(self):
        return self._kind == VARIABLE

    def is_constant(self):
        return self._kind == CONSTANT

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def same_kind(self, arg):
        """
        Check if the arguments is same
        :param arg:
        :return:
        """
        return self._kind == arg._kind

    def equals(self, arg):
        """
        Checks if the arg and self are same argument
        :param arg:
        :return:
        """
        return (self.same_kind(arg) and self._name == arg._name)

    @classmethod
    def make_var(cls, name):
        return Argument(name, VARIABLE)

    @classmethod
    def make_const(cls, name):
        return Argument(name, CONSTANT)


class Predicate(object):
    """
    Represents a predicate like P(x, y) or -P(x, y)
    Order of the args and constants matters
    For a predicate P(x, a, y)
    Name: P, Args: x, a, y; x is a var, a is constant and y is a var
    """
    def __init__(self, name, args, negative=False):
        """

        :param name: string
        :param args: Argument objects
        :param negative: bool
        """
        self._name = name
        self._args = args
        self._negative = negative

    def __str__(self):
        s = "{0}({1})".format(self._name, ",".join([arg._name for arg in self._args]))
        if self._negative:
            return "-"+s

        return s

    def get_name(self):
        return self._name

    def get_args(self):
        return self._args

    def get_negative(self):
        return self._negative

    def same_formula(self, obj):
        """
        Checks whether the self and obj has the same for i.e. P(x,y)
        has same form as P(x,y) and -P(x,y), or P(y, x), or P(a, x)
        :param obj: Predicate()
        :return: bool
        """
        if self._name != obj._name:
            return False

        if len(self._args) != len(obj._args):
            return False

        return True

    def complement_of(self, obj):
        """
        Checks whether the
        :param obj: Predicate()
        :return:
        """
        return (self.same_formula(obj) and self._negative != obj._negative)

    def same_args(self, obj):
        """
        Checks if the args in self and obj are same
        :param obj: Predicate()
        :return:
        """
        for i in range(0, len(obj._args)):
            if not self._args[i].equals(obj._args[i]):
                return False

        return True

    def equals(self, obj):
        """
        Return true if both objects are P(x,y) and P(x,y)
        :param obj: Predicate()
        :return:
        """
        if not self.same_formula(obj):
            return False

        if not self.same_args(obj):
            return False

        if self._negative != obj._negative:
            return False

        return True

    def same_predicate(self, obj):

        if self._name != obj._name:
            return False

        return True

    def complement_of_predicate(self, obj):

        if self.same_predicate(obj) == True and self._negative != obj._negative:
            return True

        return False


def unification(p1, p2, replacements):
    """
    Takes a two predicates and tries for unifying them which are same, i.e. P(x1, y1) and -P(x1, y1)
    returns None is couldn't be done else returns the list with unification, and,
    a dict() with replacements
    :param replacements:
    :return: unifiable predicates, p1, p2 and bool if unification could be done or not
    """
    p1_args = list(p1.get_args())
    p2_args = list(p2.get_args())

    if len(p1_args) != len(p2_args):
        return p1, p2, False

    if p1.same_args(p2): # return as it is
        return p1, p2, True

    for i in range(0, len(p1_args)):
        p1_arg = p1_args[i]
        p2_arg = p2_args[i]

        if p2_arg.equals(p1_arg):
            continue

        if p1_arg.is_variable() and p2_arg.is_variable():  # Replace p2 by p1
            token = replacements.get(p2_arg.get_name(), '')
            if token == '':
                token = p1_arg.get_name()
                replacements[p2_arg.get_name()] = token

            p1_args[i].set_name(token)
            p2_args[i].set_name(token)

            continue

        const = ''
        var = ''
        if p1_arg.is_constant() and p2_arg.is_variable():
            const = p1_arg.get_name()
            var = p2_arg.get_name()
        else:
            const = p2_arg.get_name()
            var = p1_arg.get_name()

        if '({0})'.format(const) in var: # can't to unification
            return p1, p2, False

        replacements[var] = const
        p1_args[i].set_name(const)
        p2_args[i].set_name(const)

    p1._args = p1_args
    p2._args = p2_args

    return p1, p2, True


def resolution(l):
    if len(l) == 1:
        return False
    setofSupport = list(l)
    fgh = list(l)
    sizeFgh = len(fgh)
    l.sort(key = len)
    for (x, y) in enumerate(l):
        refSet = l[x]
        setofSupport.remove(refSet)
        newClause = addNewClause(refSet,setofSupport)
        if newClause ==None:
            return False
        elif len(newClause) ==0:
            return True
        else:
            if newClause not in fgh:
                fgh.append(setofSupport)

    newFghSize = len(fgh)
    if newFghSize > sizeFgh:
        resolution(fgh)
    return False


def addNewClause(refSet,setofSupport):
    newClause =[]
    for (i, e) in enumerate(refSet):
        setofSupport.sort(key=len)
        for (j,k) in enumerate(setofSupport):
            check = False
            for(m,n) in enumerate(k):
                if  e.complement_of_predicate(n) or check == True:
                    check = True
                    reps = dict()
                    p1, p2, flag = unification(e, n, reps)
                    if flag == True:
                        n = p2
                        e = p1
                    else:
                        continue
            for (a,b) in enumerate(k):
                if e.complement_of(b):
                    newClause = (setofSupport[j])
                    newClause.remove(b)
                    for (c,d) in enumerate(refSet):
                        if d==e:
                            continue
                        elif d in newClause:
                            continue
                        else:
                            newClause.append(d)
                    return newClause
                else:
                    continue

    return None


def get_args_from_nodes(nodes):
    args = list()
    for node in nodes:
        if node.get_element_type() == 'function':
            symbols = node.get_child_nodes()
            arg_token = ','.join([x.get_element_value() for x in symbols])
            const = '{0}({1})'.format(node.get_element_value(), arg_token)
            args.append(Argument.make_const(const))
            continue

        if node.get_element_type() == 'symbol':
            val = node.get_element_value()
            args.append(Argument.make_const(val))
            continue

        if node.get_element_type() == 'variable':
            val = node.get_element_value()
            args.append(Argument.make_var(val))
            continue

    return args


def and_to_clausal(and_node):
    """
    Recursively call to convert a CNF tree into the clauses
    :param and_node:
    :return: list of predicates
    """
    or_list = and_node.get_child_nodes()
    clauses = list()
    for r in or_list:
        for node in r.get_child_nodes():
            # there's only one predicate here, get that
            if node.get_element_type() == 'op' and node.get_element_value() == 'NOT':
                predicate = node.get_child_nodes()[0]
                args = get_args_from_nodes(predicate.get_child_nodes())
                p = Predicate(predicate.get_element_value(), args, True)
                clauses.append(p)
            else:
                args = get_args_from_nodes(node.get_child_nodes())
                p = Predicate(node.get_element_value(), args)
                clauses.append(p)

    return clauses

test_cases = [
    ["(FORALL x (IMPLIES (p x) (q x)))", "(p (f a))", "(NOT (q (f a)))"],  # this is inconsistent
    ["(FORALL x (IMPLIES (p x) (q x)))", "(FORALL x (p x))", "(NOT (FORALL x (q x)))"],  # this is inconsistent
    ["(EXISTS x (AND (p x) (q b)))", "(FORALL x (p x))"],  # this should NOT lead to an empty clause
    ["(NOT (NOT (p a)))"],  # this should NOT lead to an empty clause
    ["(big_f (f a b) (f b c))",
     "(big_f (f b c) (f a c))",
      "(FORALL x (FORALL y (FORALL z (IMPLIES (AND (big_f x y) (big_f y z)) (big_f x z)))))",
      "(NOT (big_f (f a b) (f a c)))"], # this is inconsistent
    ["(NOT (FORALL x (EXISTS y (AND (IMPLIES (p x) (NOT (NOT (q y)))) (FORALL w (EXISTS u (OR (s w u) (NOT (NOT (t w u))))))))))"],
    ["(FORALL x (IMPLIES (AND (OR (EXISTS y (p y a b c)) (q a b)) (p x y)) (r x)))"],
    ["(FORALL x (EXISTS y (EXISTS z (AND (AND (AND (AND (l x y) (l y z)) (r z)) (IMPLIES (p z) (r z))) (IMPLIES (r z) (p z))))))", "(FORALL x (FORALL y (FORALL z (AND (EXISTS x (FORALL y (NOT (AND (p y) (l x y))))) (IMPLIES (AND (l x y) (l y z)) (l x z))))))"],
    ["(FORALL x (EXISTS y (p x y)))", "(EXISTS x (FORALL y (NOT (p x y))))"],
    ["(FORALL x (OR (NOT (p a)) (q a)))", "(FORALL x (p x))", "(OR (NOT (p (f a))) (NOT (q a)))"],
    ["(FORALL x (FORALL z (FORALL u (FORALL w (OR (p x (f x) z) (p u w w))))))", "(FORALL x (FORALL y (FORALL z (OR (NOT (p x y z)) (NOT (p z z z))))))"]
]

print(findIncSet(test_cases))
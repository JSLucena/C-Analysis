#!/usr/bin/env python3
""" A very stupid syntatic analysis, that only checks for assertion errors.
"""




import os
import sys, logging

l = logging
l.basicConfig(level=logging.DEBUG)

(name,) = sys.argv[1:]

import re
from pathlib import Path

# Read the method_name
RE = r"(?P<class_name>.+)\.(?P<method_name>.*)\:\((?P<params>.*)\)(?P<return>.*)"
if not (i := re.match(RE, name)):
    l.error("invalid method name: %r", name)
    sys.exit(-1)

TYPE_LOOKUP = {
    "Z": "boolean",
    "I": "int",
}


srcfile = (Path("src/main/java") / i["class_name"].replace(".", "/")).with_suffix(
    ".java"
)

import tree_sitter
import tree_sitter_java

JAVA_LANGUAGE = tree_sitter.Language(tree_sitter_java.language())
parser = tree_sitter.Parser(JAVA_LANGUAGE)

with open(srcfile, "rb") as f:
    l.debug("parse sourcefile %s", srcfile)
    tree = parser.parse(f.read())

simple_classname = i["class_name"].split(".")[-1]

# To figure out how to write these you can consult the
# https://tree-sitter.github.io/tree-sitter/playground
class_q = JAVA_LANGUAGE.query(
    f"""
    (class_declaration 
        name: ((identifier) @class-name 
               (#eq? @class-name "{simple_classname}"))) @class
"""
)

for node in class_q.captures(tree.root_node)["class"]:
    break
else:
    l.error(f"could not find a class of name {simple_classname} in {srcfile}")
    sys.exit(-1)

l.debug("Found class %s", node.range)

method_name = i["method_name"]

method_q = JAVA_LANGUAGE.query(
    f"""
    (method_declaration name: 
      ((identifier) @method-name (#eq? @method-name "{method_name}"))
    ) @method
"""
)

for node in method_q.captures(node)["method"]:
    if not (p := node.child_by_field_name("parameters")):
        l.debug(f"Could not find parameteres of {method_name}")
        continue

    params = [c for c in p.children if c.type == "formal_parameter"]
    if len(params) == len(i["params"]) and all(
        (tp := t.child_by_field_name("type")) is not None
        and tp.text is not None
        and TYPE_LOOKUP[tn] == tp.text.decode()
        for tn, t in zip(i["params"], params)
    ):
        break
else:
    l.warning(f"could not find a method of name {method_name} in {simple_classname}")
    sys.exit(-1)

l.debug("Found method %s %s", method_name, node.range)

body = node.child_by_field_name("body")
assert body and body.text
for t in body.text.splitlines():
    l.debug("line: %s", t.decode())





###### ASSERT QUERIES###########################
assert_false_q = JAVA_LANGUAGE.query(
    f"""(assert_statement
	(false) @false
) @assert""")

assert_true_q = JAVA_LANGUAGE.query(
    f"""(assert_statement
	(true) @true
) @assert""")

assert_binary_expr = JAVA_LANGUAGE.query(
    f"""(assert_statement
	(binary_expression (expression) @left (expression) @right ) @bin
) @assert
""")

assert_id = JAVA_LANGUAGE.query(f"""
    (assert_statement
	(identifier) @name
) @assert
""")

assert_inside_for_q = JAVA_LANGUAGE.query(f"""
    (for_statement
    init: (_) @for_init
    condition: (expression) @for_condition
    update: (_) @for_update
    body: (block
      (assert_statement) @assert_inside_for
    )
  )
""")
assert_inside_while_q = JAVA_LANGUAGE.query(f"""
    (while_statement
    condition: (parenthesized_expression) @for_condition
    body: (block
      (assert_statement) @assert_inside_for
 ))
""")

assert_after_for_q = JAVA_LANGUAGE.query(f"""
    ((for_statement
    init: (_) @for_init
    condition: (expression) @for_condition
    update: (_) @for_update
    body: (block)
      
    )
    (assert_statement) @assert_inside_for)
""")
assert_after_while_q = JAVA_LANGUAGE.query(f"""
    ((while_statement
    condition: (parenthesized_expression) @for_condition
    body: (block) @block)
    (assert_statement) @assert_inside_for )
""")
###############################################


##### Variable Declarations ###################
var_declaration_q = JAVA_LANGUAGE.query(f"""
(local_variable_declaration
        type: (_) @type 
        declarator: (variable_declarator 
          (identifier) @name
          (expression) @val
        ) @declarator
    ) @declaration
""")
array_declaration_q = JAVA_LANGUAGE.query(f"""
(local_variable_declaration
        type : (_) @type 
        declarator: (
          variable_declarator (identifier) @name
          (array_initializer) @val 
        ) @declarator
    ) @declaration
""")

assignment_expression_q = JAVA_LANGUAGE.query(f"""
(assignment_expression
    left: (identifier) @var_name
    right: (_) @assigned_value
) @assignment
""")
##################################################

##### DIVISION BY ZERO############################
division_by_literal_zero = JAVA_LANGUAGE.query(f"""
(binary_expression
    left: (_) @numerator
    operator: "/"
    right: (decimal_integer_literal) @divisor
        (#eq? @divisor "0")
) @division_by_zero
 
""")

division_by_variable = JAVA_LANGUAGE.query(f"""
(binary_expression
    left: (_) @numerator
    operator: "/"
    right: (identifier) @divisor
) @division_by_zero
 
""")

variable_zero_assignment = JAVA_LANGUAGE.query(f"""
(local_variable_declaration
    (variable_declarator
        name: (identifier) @var_name
        value: (decimal_integer_literal) @value
        (#eq? @value "0"))
) @assignment
""")
###################################################
############## OUT OF BOUNDS QUERIES###############
array_access_q = JAVA_LANGUAGE.query(f"""
(array_access
    array: (identifier) @array_name
    index: (_) @index
) @array_access
""")
###################################################
######### SOME VARIABLES#############################
left = None
right = None
bin = None
has_parameter = False
has_variable = False
has_array = False
typ = None
found_oob = False
found_assert_error = False
found_0_div = False
found_null_ptr = False
#####################################################

def checkParamOperations(t, val):
    if ">" in t:
        val = int(val)
        prob = (float(sys.maxsize) - val) / (float(sys.maxsize) * 2) *100
        print("assertion error;" + str(int(prob)) + "%")

    elif "!=" in t:
        l.debug("Very improbable of assertion being true")
        print("assertion error;15%")



def checkVariableOperations(t,var, val):
    length = None
    if "length" in t:
        length = var.count(",") + 1
    if "==" in t:
        if length != None and int(val) == length:
            l.debug("Assertion is correct")
            print("assertion error;5%")

            return
        if var == val:
                l.debug("Assertion is correct")
                print("assertion error;5%")

                return
        else:
            l.debug("Assertion is wrong")
            print("assertion error;95%")

    if ">" in t:
        if var[-1] > val:
            l.debug("Assertion is correct")
            print("assertion error;5%")

        else:
            l.debug("Assertion is wrong")
            print("assertion error;95%")


def checkObjectMethodo(t, obj,right):
    checkVariableOperations(t,obj[-1],right)


def detectInfiniteLoops(body):
    remove_things = False
    count = 0
    for node in body.children:
        if node.type == "while_statement":
            for inner in node.children:
                if inner.type == "parenthesized_expression":
                    first = inner.children[1].children[0].text.decode()
                    op = inner.children[1].children[1].text.decode()
                    right = inner.children[1].children[2].text.decode()
                    for node2, t in var_declaration_q.captures(body).items():
                        if node2 == "name":
                            if first == t[0].text.decode():
                                first = [first,1,typ] #1 here represents a variable
                            if right == t[0].text.decode():
                                right = [right,1,typ]
                        if node2 == "val":
                                if op == ">":
                                    if int(t[0].text.decode()) > int(right):
                                        for node3 , t2 in assignment_expression_q.captures(node).items():
                                            if node3 == "name":
                                                if first[0] == t2[0].text.decode():
                                                    l.debug("condition is changed inside the loop, may not loop forever")
                                                    return
                                        remove_things = True
                                        break
        
        if remove_things:
            copy = body
            idx = body.children.index(node)
           # print(idx)
            for i in range(idx+1, copy.child_count-1):
                copy.children.pop(i)
            #print(body.children)
            return copy
        count += 1 


#####################################################

#body = detectInfiniteLoops(body)

for node in assert_true_q.captures(body).items():
    l.debug("Assertion will be true")
    print("assertion error;15%")
    found_assert_error = True
    break

for node in assert_false_q.captures(body).items():
    l.debug("Assertion is false")
    print("assertion error;80%")
    found_assert_error = True
    break

for node, t in assert_id.captures(body).items():
    l.debug("Assert is an identifier, performing additional check")
    if node == "name":
        node = t[0].text.decode()
        l.debug(node)
        for parameter in params:
            if node == parameter.children[1].text.decode():
                l.debug("Assert target is an argument, cannot predict")
                print("assertion error;50%")
                found_assert_error = True
                break


for node, t in assert_binary_expr.captures(body).items():
    if node == "bin":
        bin = t[0].text.decode()
    if node == "left":
        left = t[0].text.decode()
    if node == "right":
        right = t[0].text.decode()

if bin != None:
    l.debug("Asserting binary expression")
    for parameter in params:
        if left == parameter.children[1].text.decode():
            l.debug("Left expression is an argument")
            left = [left, 0] #parameter
        if right == parameter.children[1].text.decode():
            l.debug("Right expression is an argument")
            right = [right, 0] #parameter
    for node, t in array_declaration_q.captures(body).items():
        if node == "name":
            if "." in left:
                split = left.split(".")
                if t[0].text.decode() == split[0]:
                    left = [split[0],2,typ, split[1]] #2 here represents an object
            else:
                if left == t[0].text.decode():
                    left = [left,2,typ]
        if node == "type":
            typ = t[0].text.decode()
        if node == "val":
            if type(left) is list:
                left.append(t[0].text.decode())
    for node, t in var_declaration_q.captures(body).items():
        if node == "type":
            typ = t[0].text.decode()
        if node == "name":
            if left == t[0].text.decode():
                left = [left,1,typ] #1 here represents a variable
            if right == t[0].text.decode():
                right = [right,1,typ]
        if node == "val":
            if len(left) == 3:
                left.append(t[0].text.decode()) #1 here represents a variable
            if len(right) == 3:
                right.append(t[0].text.decode())

if type(left) is list:
    if left[1] == 0:
        if type(right) is not list:
            checkParamOperations(bin,right)
            found_assert_error = True
    elif left[1] == 1:
        if type(right) is not list:
            checkVariableOperations(bin,left,right)
            found_assert_error = True
    elif left[1] == 2:
        if left[-1] == "null":
            print("null pointer;80%")
            found_null_ptr = True
        else:
            if type(right) is not list:
                checkObjectMethodo(bin,left,right)
                found_assert_error = True




for node, t in division_by_literal_zero.captures(body).items():
    if node == "division_by_zero":
        l.debug("Found divison by zero")
        print("divide by zero;85%")
        found_0_div = True

for node, t in division_by_variable.captures(body).items():
    l.debug("Divisor is a variable, performing additional checks")
    if node == "divisor":
        node = t[0].text.decode()
        l.debug(node)
        for parameter in params:
            if node == parameter.children[1].text.decode():
                l.debug("Assert target is an argument")
                l.debug("Low chance of it being zero")
                print("divide by zero;15%")
                found_0_div = True
                break


for node, t in array_access_q.captures(body).items():
    if node == "array_name":
        bin = t[0].text.decode()
    if node == "index":
        for node2, t2 in array_declaration_q.captures(body).items():
            if node2 == "name":
                left = t2[0].text.decode()
            if node2 == "val":
                if t2[0].text.decode() == "null":
                    print("null pointer;80%")
                    found_null_ptr = True
                    break
                size = t2[0].text.decode().count(",") + 1
                try:
                    idx = int(t[0].text.decode())
                except:
                    l.debug("oops, index is a variable")
                    break
                if left == bin:
                    if idx < 0 or idx >= size:
                        print("out of bounds;80%")
                        found_oob = True
        if left != bin:
            l.debug("Oops, the array does not exist")
            print("null pointer;80%")
            found_null_ptr = True
            break
                        

"""
for node, t in assert_q.captures(body).items():
    if t == "assert":
        break
else:
    l.debug("Did not find any assertions")
    print("assertion error;20%")
    sys.exit(0)

l.debug("Found assertion")
print("assertion error;80%")
"""

if not found_oob:
    print("out of bounds;30%")
if not found_assert_error:
    print("assertion error;30%")
if not found_0_div:
    print("divide by zero;30%")
if not found_null_ptr:
    print("null pointer;30%")
sys.exit(0)

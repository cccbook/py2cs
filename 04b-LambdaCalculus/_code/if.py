IF = lambda cond:lambda job_true:lambda job_false:job_true if cond else job_false

print(IF(True)("Yes")("No"))
print(IF(False)("Yes")("No"))

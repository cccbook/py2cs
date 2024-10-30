import dis

for op in range(len(dis.opname)):
    if dis.opname[op] is not None:
        has_arg = op >= dis.HAVE_ARGUMENT
        print(f"{dis.opname[op]}: {'Has argument' if has_arg else 'No argument'}")

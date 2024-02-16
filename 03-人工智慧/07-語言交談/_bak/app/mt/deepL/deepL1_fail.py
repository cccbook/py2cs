import deepl
# source = 'The number of micro operations is minimized without impacting the quality of the generated code much. For example, instead of generating every possible move between every 32 PowerPC registers, we just generate moves to and from a few temporary registers. These registers T0, T1, T2 are typically stored in host registers by using the GCC static register variable extension.'
source = 'The number of micro operations is minimized without impacting the quality of the generated code much. '
# source = 'Hello, How are you!'
target = deepl.translate(source_language="EN", target_language="ZH", text=source)
print('================ 原文 ===================')
print(source)
print('================ 譯文 ===================')
print(target)

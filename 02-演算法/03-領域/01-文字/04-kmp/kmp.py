def compute_lps(pat): 
    M = len(pat)
    len0 = 0 
    lps = [0]*M
    i = 1
    while i < M: 
        if pat[i]== pat[len0]: 
            len0 += 1
            lps[i] = len0
            i += 1
        else: 
            if len0 != 0:
                len0 = lps[len0-1] 
            else: 
                lps[i] = 0
                i += 1
    return lps

def kmp_search(pat, txt): 
    M = len(pat)
    N = len(txt) 
    j = 0 
    lps = compute_lps(pat) 
  
    i = 0 
    while i < N: 
        if pat[j].lower() == txt[i].lower(): 
            i += 1
            j += 1
  
        if j == M: 
            return (i-j) 
            j = lps[j-1] 
  
        elif i < N and pat[j] != txt[i]: 
            if j != 0: 
                j = lps[j-1] 
            else: 
                i += 1

print('kmp_search("ABABCABAB","ABABDABACDABABCABAB")=', kmp_search("ABABCABAB","ABABDABACDABABCABAB"))

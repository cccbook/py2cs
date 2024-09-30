def pair(x, y):
    def dispatch(m):
        return (x if m == 0 else
                y if m == 1 else
                None)

    return dispatch

def head(z): return z(0)

def tail(z): return z(1)

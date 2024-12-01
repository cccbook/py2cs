// https://gist.github.com/Avaq/1f0636ec5c8d6aed2e45?permalink_comment_id=3908358

const I  = x => x
const K  = x => y => x
const A  = f => x => f (x)
const T  = x => f => f (x)
const W  = f => x => f (x) (x)
const C  = f => y => x => f (x) (y)
const B  = f => g => x => f (g (x))
const S  = f => g => x => f (x) (g (x))
const S_ = f => g => x => f (g (x)) (x)
const S2 = f => g => h => x => f (g (x)) (h (x))
const P  = f => g => x => y => f (g (x)) (g (y))
const Y  = f => (g => g (g)) (g => f (x => g (g) (x)))
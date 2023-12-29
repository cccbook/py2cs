from Bio.Seq import Seq
from Bio.SeqUtils import GC

my_seq = Seq('GATCGATGGGCCTATATAGGATCGAAAATCGC')
print('my_seq=', my_seq)
print('len(my_seq)=', len(my_seq))
print('my_seq.count("G")=', len(my_seq))
print('MY_GC(my_seq)=100 * float(my_seq.count("G") + my_seq.count("C")) / len(my_seq)=', 100 * float(my_seq.count("G") + my_seq.count("C")) / len(my_seq))
print('GC(my_seq)=', GC(my_seq))

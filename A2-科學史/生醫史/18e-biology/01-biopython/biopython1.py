from Bio.Seq import Seq
s = Seq("AGTACACTGGT")
print('s=', s)
print('s.complement()=', s.complement())
print('s.reverse_complement()=', s.reverse_complement())

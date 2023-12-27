from Bio.Seq import Seq
from Bio.SeqUtils import GC

coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
print('coding_dna=', coding_dna)
print('coding_dna.reverse_complement()=', coding_dna.reverse_complement())
print('coding_dna.transcribe()=', coding_dna.transcribe())
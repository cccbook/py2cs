from Bio.Blast import NCBIWWW

sequence = "ATGCAGCTAGCTAGCTACGATCGATCAGCTACATCGACTAGCTACGATCG"
result_handle = NCBIWWW.qblast("blastn", "nt", sequence)
print(result_handle)

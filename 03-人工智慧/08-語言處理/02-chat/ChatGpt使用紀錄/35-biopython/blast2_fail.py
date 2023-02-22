# 來源 -- https://homes.cs.washington.edu/~ruzzo/courses/gs559/10wi/data/blast-demo.py
'''
   Demo of network Blast access via Biopython
'''

import os
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
from Bio import Entrez
Entrez.email = "ImaStudent@uw.edu"  # <=== put your real email address here!

# Code below sends blast query to NCBI and *caches* the result (i.e., saves it locally),
# to avoid repeated queries as you debug subsequent code.  Delete the cache file 
# ("trnablast-cache.xml") and rerun to force the query to be reissued.

cacheFileName = "trnablast-cache.xml"
if(not os.path.exists(cacheFileName)):

	# my query is 1st +strand, short Mj tRNA (#7, 190831..190905  74nt):
	query = "GGGGCCGTGGGGTAGCCTGGATATCCTGTGCGCTTGGGGGGCGTGCGACCCGGGTTCAAGTCCCGGCGGCCCCA"

	# qblast has 43 parameters; you'll always want the 1st three:
	#  - which Blast program (e.g., "blastn"),
	#  - which database (e.g., nr="nonredundant"), and 
	#  - the query sequence itself.  	
	# All others are optional "keyword parameters", i.e., can be specified "by name" so that you
	# don't have to memorize their exact order and position in a long parameter list.  The following
	# "entrez_query" param limits blast to sequences from a specific organism:

	myEntrezQuery = "Methanocaldococcus jannaschii[Organism]"

	prog = 'blastn'
	db = 'refseq_genomic'
	# Blast it:
	print('Issuing net Blast query')
	res_handle = NCBIWWW.qblast(prog, db, query, entrez_query = myEntrezQuery)
	
	# save the result:
	savefile = open(cacheFileName, "w")
	savefile.write(res_handle.read())
	savefile.close()
	res_handle.close()

# Get cached result:
print('Fetching cached Blast resuts')
resultHandle = open(cacheFileName, "r")
blastRecord = NCBIXML.read(resultHandle)

# Print a representative hit:
print(blastRecord.alignments[0].hsps[0])

####
#
# EXERCISE:
#   Find data, in blastRecord, giving: score, Evalue, alignment length & start coordinate
#   of the first high-scoring-pair of the first alignment reported by Blast. 
#   Hint: use dir(), help(), type(), str(), print(), ...
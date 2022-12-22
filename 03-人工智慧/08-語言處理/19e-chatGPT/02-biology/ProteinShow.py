# 導入 Biopython 的 PDB 模組
from Bio.PDB import PDBParser

# 讀取 PDB 格式的文件
parser = PDBParser()
structure = parser.get_structure('1GZM', './1gzm.pdb')

# 獲取結構單位
model = structure[0]

# 獲取殘基
residues = [res for res in model.get_residues()]

# 獲取原子
atoms = [atom for atom in residues[0].get_atoms()]

# 畫出蛋白質結構
from Bio.PDB.vectors import calc_angle, calc_dihedral

v1 = atoms[0].get_vector()
v2 = atoms[1].get_vector()
v3 = atoms[2].get_vector()

angle = calc_angle(v1, v2, v3)
print(angle)

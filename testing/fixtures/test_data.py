SIMPLE_TOKENS = {t: i for i, t in enumerate(['$', '^', '(', ')', '1', '2', '3', '=', 'C', 'F', 'N', 'O', 'S', 'c', 'n'])}

PARACETAMOL = "CC(=O)NC1=CC=C(C=C1)O"
SCAFFOLD_SUZUKI = 'Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)[*]'

INVALID = 'INVALID'
NONSENSE = 'C1CC(Br)CCC1[ClH]'
ASPIRIN='O=C(C)Oc1ccccc1C(=O)O'
CELECOXIB='O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
IBUPROFEN='CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O'
ETHANE = 'CC'
PROPANE='CCC'
BUTANE='CCCC'
PENTANE = 'CCCCC'
HEXANE = 'CCCCCC'
METAMIZOLE='CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)CS(=O)(=O)O'
CAFFEINE='CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
COCAINE='CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC'
BENZENE='c1ccccc1'
TOLUENE='c1ccccc1C'
ANILINE='c1ccccc1N'
AMOXAPINE = 'C1CN(CCN1)C2=NC3=CC=CC=C3OC4=C2C=C(C=C4)Cl'
GENTAMICIN = 'CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC'
METHOXYHYDRAZINE = 'CONN'
HYDROPEROXYMETHANE = 'COO'

WARHEAD_PAIR = '*C1CCCCC1|*C1CCCC(ON)C1'

IBUPROFEN_TOKENIZED = ["^", "C", "C", "(", "C", ")", "C", "c", "1", "c", "c", "c", "(", "c", "c", "1", ")", "[C@@H]",
                       "(", "C", ")", "C", "(", "=", "O", ")", "O", "$"]
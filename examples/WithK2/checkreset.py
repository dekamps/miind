with open('rinzel.res') as res_file:
	for l_check1 in res_file:
		tokens1 = l_check1.split('\t')
		if len(tokens1) != 3:
			continue
		froms = tokens1[0].split(',')
		with open('rinzel.res') as res_file2:
			for l_check2 in res_file2:
				tokens2 = l_check2.split('\t')
				if len(tokens2) != 3:
					continue
				tos = tokens2[1].split(',')
				if froms[0] == tos[0] and froms[1] == tos[1]:
					print froms

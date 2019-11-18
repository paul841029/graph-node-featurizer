snuba_exp:
	python generate_features.py\
		--example ${example}\
		--dataset ${data}\
		--gt /luh/synthesis_data_snuba/ground_truth/${gt_level}/${data}.json
	python snuba.py\
		--num_example ${example}\
		--dataset ${data}\
		--gt ${gt_level}

.PHONY: split_files
split_files:
	rm /luh/synthesis_data_snuba/${data}/example/total/train/*.db
	rm /luh/synthesis_data_snuba/${data}/example/total/test/*.db
	rm /luh/synthesis_data_snuba/${data}/example/total/validation/*.db
	python files_spliter.py\
		--input /luh/synthesis_data/${data}/example/total\
		--train /luh/synthesis_data_snuba/${data}/example/total/train\
		--test /luh/synthesis_data_snuba/${data}/example/total/test\
		--val /luh/synthesis_data_snuba/${data}/example/total/validation
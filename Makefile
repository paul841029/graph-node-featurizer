snuba_exp:
	python generate_features.py\
		--dataset ${data}\
		--example ${example}\
		--rate ${rate}
	python snuba.py\
		--dataset ${data}\
		--gt mix\
		--example ${example}

.PHONY: split_files
split_files:
	rm -fr /luh/synthesis_data_snuba/${data}/example/total/train/
	rm -fr /luh/synthesis_data_snuba/${data}/example/total/test/
	# rm -f /luh/synthesis_data_snuba/${data}/example/total/validation/*.db
	python files_spliter.py\
		--input /luh/synthesis_data/${data}/example/total\
		--train /luh/synthesis_data_snuba/${data}/example/total/train\
		--test /luh/synthesis_data_snuba/${data}/example/total/test\
		--data ${data}
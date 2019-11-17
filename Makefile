snuba_exp:
	python generate_features.py\
		--example ${example}\
		--dataset president_mix\
		--gt /luh/synthesis_data/president_mix/example/gt/from_table/president_wiki_cell_example.json
	python snuba.py\
		--num_example ${example}\
		--dataset presdient_mix
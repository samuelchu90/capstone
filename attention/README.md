open cmd and run the following commands:

cd data

type AMPs.fa Non-AMPs.fa > combined.txt

cd ..

perl script/format.pl data/combined.txt none > input.txt

python script/prediction_attention.py input.txt output_attention.txt

you can check the results in attention_result.ipynb
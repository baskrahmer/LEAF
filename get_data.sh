mkdir data

wget https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz
gunzip openfoodfacts-products.jsonl.gz
mv openfoodfacts-products.jsonl data/products.jsonl

curl -o ciqual.csv "https://data.ademe.fr/data-fair/api/v1/datasets/agribalyse-31-synthese/lines?size=10000&page=1&format=csv"
mv ciqual.csv data/

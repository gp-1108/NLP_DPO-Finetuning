git pull

docker build -t dpo_webapp:thesis -f dataset_visualization/Dockerfile .

docker run -d -v /certs:/certs -p 5005:5005 dpo_webapp:thesis
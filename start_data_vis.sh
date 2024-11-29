# Updating the repo
git pull

# Stopping container running dpo_webapp:thesis
docker stop $(docker ps -a -q --filter "ancestor=dpo_webapp:thesis")

# Removing container running dpo_webapp:thesis
docker rm $(docker ps -a -q --filter "ancestor=dpo_webapp:thesis")

# Rebuiding the docker image
docker build -t dpo_webapp:thesis -f dataset_visualization/Dockerfile .

# Running the docker container
docker run -d -v /certs:/certs -p 5005:5005 dpo_webapp:thesis
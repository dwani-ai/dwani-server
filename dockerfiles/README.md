
docker build -f Dockerfile.base -t slabstech/dhwani-server-base .

docker push slabstech/dhwani-server-base  


docker build -f Dockerfile -t slabstech/dhwani-server-model .

docker push slabstech/dhwani-model-server

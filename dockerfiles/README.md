
docker build -f Dockerfile.base -t slabstech/dhwani-server-base .

docker push slabstech/dhwani-server-base  

 docker build -f Dockerfile.models --build-arg HF_TOKEN_DOCKER=$HF_TOKEN -t slabstech/dhwani-server-models .

 ocker push slabstech/dhwani-server-models 

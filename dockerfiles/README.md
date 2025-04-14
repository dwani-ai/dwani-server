Building Docker Images for Dhwani 

- Model Server 

  - Base Model
    - docker build -f Dockerfile.base -t slabstech/dhwani-server-base .
    - docker push slabstech/dhwani-server-base  

  - Deploy model
    - docker build -f Dockerfile -t slabstech/dhwani-server-model .
    - docker push slabstech/dhwani-model-server


- API Server
  - Base model
    - docker build -f Dockerfile.base -t slabstech/dhwani-api-server-base .
    - docker push slabstech/dhwani-api-server-base

  - Deploy model
    - docker build -f Dockerfile -t slabstech/dhwani-api-server .
    - docker push slabstech/dhwani-api-server

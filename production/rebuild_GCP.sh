PROJECT_ID=p-g-us-adv-x-dat-aia-proto-1
COMPUTE_ZONE=us-central1-a
CLUSTER_NAME=lego-demo

# To set a default project, run the following command from Cloud Shell:
gcloud config set project ${PROJECT_ID}
    
# To set a default compute zone, run the following command:
gcloud config set compute/zone ${COMPUTE_ZONE}

# Get kubernete cluster credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${COMPUTE_ZONE}

DOCKER_IMAGE_NAME=lego
DOCKER_VERSION=latest
docker tag ${DOCKER_IMAGE_NAME} gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}:${DOCKER_VERSION}
docker push gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}:${DOCKER_VERSION}

kubectl delete deployment lego-demo
kubectl create -f modelhive.yaml


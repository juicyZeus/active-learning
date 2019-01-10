#! /bin/sh

docker container rm -f activenlp
docker image rm tax-active-learning
docker build -t tax-active-learning .
docker run -d --name activenlp -v saltitx_data:/app/saltitx_data -p 80:80 tax-active-learning

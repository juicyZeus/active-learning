#! /bin/sh
docker container rm -f lego_box
#docker image rm news-radar
docker build -f Dockerfile -t lego .
#docker run -d --name news-radar-container -p 80:80 -p 8888:8888 news-radar
#docker run -it --name news-radar-container -p 80:80 -p 8888:8888 --entrypoint "jupyter lab --ip=0.0.0.0 --allow-root" news-radar

docker run -it --name lego_box -p 9001:80 -p 9999:8888 --entrypoint /bin/bash lego
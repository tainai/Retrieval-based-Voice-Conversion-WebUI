aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 075389491675.dkr.ecr.ap-northeast-2.amazonaws.com
docker buildx build --platform=linux/amd64 -f Dockerfile_api -t rvc-api:latest .
docker tag rvc-api:latest 075389491675.dkr.ecr.ap-northeast-2.amazonaws.com/rvc-api:latest
docker push 075389491675.dkr.ecr.ap-northeast-2.amazonaws.com/rvc-api:latest

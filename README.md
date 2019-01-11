# DeepLearningLab
In this repository, you can find some samples and experiments about machine and deep learning, specially about image processing and objects recognition.

Use Docker to run the project:
```sh
docker pull sergiogq/deeplearninglab
docker run -it -p 8888:8888 -p 6006:6006 -v ~/:/host sergiogq/deeplearninglab jupyter notebook --allow-root /host
```

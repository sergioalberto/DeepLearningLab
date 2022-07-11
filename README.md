# DeepLearningLab
In this repository, you can find some samples and experiments about machine and deep learning, specially about image processing and objects recognition.

### Build the docker lab
```sh
docker build -t sergiogq/deeplearninglab .
```

### Use Docker to run the project:
```sh
docker pull sergiogq/deeplearninglab
```
```sh
docker run -it -p 8888:8888 -p 6006:6006 -v ~/:/host sergiogq/deeplearninglab jupyter notebook --allow-root /host
or
docker run -it -p 8888:8888 -p 6006:6006 -v ~/:/host sergiogq/deeplearninglab
```

Go to http://127.0.0.1:8888

# Installation
1) Install dependencies

```
pip3 install -r requirements.txt
```

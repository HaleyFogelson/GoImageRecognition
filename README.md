# Go Image Recognition


## Project Description
This is a program built using go that classifies animals using tensor flow and google's data set for animals. This project is for getting used to using go. Inorder to run this program you must pass in an image. 


## How to use the repository
1.Clone the project
```
git clone https://github.com/HaleyFogelson/GoImageRecognition.git
```
2.Configure Docker 


# Build the docker file
```
docker build --tag src:1.0 .
```


# How to Run the Docker file
#### Run with url address of an image
```
docker run src:1.0 url_address_of_image
```
#### Run with downloaded image
```
docker run src:1.0 path_of_image upload
```

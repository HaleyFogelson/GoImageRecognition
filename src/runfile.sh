#!/bin/bash
docker build --tag src:1.0 .
docker run src:1.0 https://dogtime.com/assets/uploads/2018/10/puppies-cover-1280x720.jpg
docker run src:1.0 https://media1.popsugar-assets.com/files/thumbor/L4cUBqWQhC4Zfnqv6e7AZ9kjUpY/0x159:2003x2162/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2019/08/07/875/n/24155406/9ffb00255d4b2e079b0b23.01360060_/i/Cute-Pictures-German-Shepherd-Puppies.jpg
docker run src:1.0 ./images/Rona.jpg upload
for f in ./images/*.jpg; do 
docker run src:1.0 $f upload
done

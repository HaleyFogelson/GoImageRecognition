FROM ctava/tfcgo
RUN mkdir -p /model && \
  curl -o /model/inception5h.zip -s "http://download.tensorflow.org/models/inception5h.zip" && \
  unzip /model/inception5h.zip -d /model
WORKDIR /go/src/imgrecognition
COPY . .
#RUN go get "github.com/stretchr/testify/assert"
RUN go build
RUN go test -v
ENTRYPOINT [ "/go/src/imgrecognition/imgrecognition" ]


package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"sort"

	"github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const (
	graphFile  = "/model/tensorflow_inception_graph.pb"
	labelsFile = "/model/imagenet_comp_graph_label_strings.txt"
)

// a structure that represents a label
type Label struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

//list of the labels
type Labels []Label

//defines how we sort all the labels
func (a Labels) Len() int           { return len(a) }
func (a Labels) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Labels) Less(i, j int) bool { return a[i].Probability > a[j].Probability }
func (a Labels) Equals(other Labels) bool {
	if a.Len() != other.Len() {
		return false
	}
	for i, l := range a {
		if l.Label != other[i].Label || l.Probability != other[i].Probability {
			return false
			log.Printf("not equal \nwant: %s, %f \ngot: %s, %f", l.Label, l.Probability, other[i].Label, other[i].Probability)
		}
	}
	return true
}
func (a Labels) getLabels() string {
	result := ""
	for _, label := range a {
		result = fmt.Sprintf("%s label: %s prob: %f,", result, label.Label, label.Probability)
	}
	return result
}

// This is the main method
func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")

	if checkArgs(os.Args) {

		response, e := http.Get(os.Args[1])
		if e != nil {
			log.Fatalf("unable to get image from url: %v", e)
		}
		defer response.Body.Close()

		//Gets the normalized graph
		graph, input, outputG, err := getNormalizedGraph()
		if err != nil {
			log.Fatalf("unable to get normalized graph %v", e)
		}

		// turns the image into a tensor so it can be comapared to by the model
		tensor, err := imageToTensor(response.Body, tensorflow.NewTensor, runSession, graph, input, outputG)
		if err != nil {
			log.Fatalf("cannot create tensor from the model %v", err)
		}

		modelGraph, labels, err := loadModel(graphFile, labelsFile)
		if err != nil {
			log.Fatalf("There was an issue loading the model %v", err)
		}

		// Create a session for it to guess what the image is based off the model
		session, err := tensorflow.NewSession(modelGraph, nil)
		if err != nil {
			log.Fatalf("There was an error initializing the session: %v", err)
		}

		output, err := session.Run(
			map[tensorflow.Output]*tensorflow.Tensor{
				modelGraph.Operation("input").Output(0): tensor,
			},
			[]tensorflow.Output{
				modelGraph.Operation("output").Output(0),
			},
			nil)
		if err != nil {
			log.Fatalf("could not make a guess: %v", err)
		}
		//gets the top 5 guesses
		res := getTopFiveLabels(labels, output[0].Value().([][]float32)[0])
		//prints out the top 5 guesses
		for _, l := range res {
			fmt.Printf("label: %s, probability: %.2f%%\n", l.Label, l.Probability*100)
		}
	} else {
		log.Fatalf("usage: imgrecognition <image_url>")
	}
}

//checks the url we are trying to search for
func checkArgs(args []string) bool {
	//the url is too short meaning it is not valid
	if len(args) < 2 {
		return false
	}
	//prints out the url we are trying to search for
	url := args[1]
	fmt.Printf("the url name we are searching for: %s\n", url)
	return true
}

// function that loads a pretrained model
func loadModel(graphFileName string, labelsFileName string) (*tensorflow.Graph, []string, error) {
	// Load inception model
	model, err := ioutil.ReadFile(graphFileName)
	if err != nil {
		return nil, nil, err
	}
	graph := tensorflow.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, nil, err
	}

	// Load labels
	labelsFile, err := os.Open(labelsFileName)
	if err != nil {
		return nil, nil, err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	return graph, labels, scanner.Err()
}

//gets the top 5 labels the image is most likely to be
func getTopFiveLabels(labels []string, probabilities []float32) Labels {
	var resultLabels []Label
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		resultLabels = append(resultLabels, Label{Label: labels[i], Probability: p})
	}

	sort.Sort(Labels(resultLabels))
	return resultLabels[:5]
}

//this is a function inorder to normalize an image by turning it into a tensor
func imageToTensor(body io.ReadCloser, createTensor func(value interface{}) (*tensorflow.Tensor, error),
	runSession func(*tensorflow.Session, *tensorflow.Tensor, tensorflow.Output,
	tensorflow.Output) ([]*tensorflow.Tensor, error), graph *tensorflow.Graph, input tensorflow.Output,
	output tensorflow.Output) (*tensorflow.Tensor, error) {
	//buffers from the body function
	var buf bytes.Buffer
	io.Copy(&buf, body)

	tensor, err := createTensor(buf.String())
	if err != nil {
		return nil, err
	}

	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	normalized, err := runSession(session, tensor, input, output)
	if err != nil {
		return nil, err
	}

	return normalized[0], nil
}

func runSession(session *tensorflow.Session, tensor *tensorflow.Tensor, input tensorflow.Output,
	output tensorflow.Output) ([]*tensorflow.Tensor, error) {
	normalized, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			input: tensor,
		},
		[]tensorflow.Output{
			output,
		},
		nil)
	return normalized, err
}

// Creates a graph to decode, rezise and normalize an image
//normalizes an image to tensor flow image
func getNormalizedGraph() (graph *tensorflow.Graph, input, output tensorflow.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tensorflow.String)
	// 3 return RGB image
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	// Sub: returns x - y element-wise
	output = op.Sub(s,
		// make it 224x224: inception specific
		op.ResizeBilinear(s,
			// inserts a dimension of 1 into a tensor's shape.
			op.ExpandDims(s,
				// cast image to float type
				op.Cast(s, decode, tensorflow.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{224, 224})),
		// mean = 117: inception specific
		op.Const(s.SubScope("mean"), float32(117)))
	graph, err = s.Finalize()

	return graph, input, output, err
}

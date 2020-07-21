package main

import (
	"fmt"
	"github.com/tensorflow/tensorflow/tensorflow/go"
	"log"
	"net/http"
	"testing"
)

func Test_checkUrl(t *testing.T) {
	t.Run("happy path", func(t *testing.T) {
	})
}

func TestCheckArgs(t *testing.T) {
	// assert := assert.New(t)
	// fmt.Printf("%t",checkArgs([]string{"first","https://cdn.pixabay.com/photo/2019/07/30/05/53/dog-4372036__340.jpg"}))
	// assert.True(checkArgs([]string{"first","https://cdn.pixabay.com/photo/2019/07/30/05/53/dog-4372036__340.jpg"}))
	// assert.False(checkArgs([]string{"first"}))
	var tests = []struct {
		name string
		args []string
		want bool
	}{
		{"not enough info", []string{"1"}, false},
		{"alright arguments", []string{"1", "www.google.com"}, true},
	}
	for _, tt := range tests {
		testname := fmt.Sprintf(tt.name)
		t.Run(testname, func(t *testing.T) {
			ans := checkArgs(tt.args)
			if ans != tt.want {
				t.Errorf("got %t, want %t", ans, tt.want)
			}
		})
	}
}

type emptyValue struct{}

type classForFunctions interface {
	runSessionMock(*tensorflow.Session, *tensorflow.Tensor, tensorflow.Output, tensorflow.Output) ([]*tensorflow.Tensor, error)
	createTensor(value interface{}) (*tensorflow.Tensor, error)
}

type mock struct {
	Tensor  *tensorflow.Tensor
	Tensors []*tensorflow.Tensor
}

func (m mock) runSessionMock(_ *tensorflow.Session, _ *tensorflow.Tensor, _ tensorflow.Output,
	_ tensorflow.Output) ([]*tensorflow.Tensor, error) {
	return m.Tensors, nil
}

func (m mock) createTensorMock(_ interface{}) (*tensorflow.Tensor, error) {
	return m.Tensor, nil
}

func TestImageToTensor(t *testing.T) {
	mockInfo1 := emptyValue{}
	tensor1, _ := tensorflow.NewTensor(mockInfo1)
	mockInfo2 := emptyValue{}
	tensor2, _ := tensorflow.NewTensor(mockInfo2)
	var tensors = []*tensorflow.Tensor{tensor1, tensor2}
	var m = mock{tensor1, tensors}
	var tests = []struct {
		name      string
		url       string
		want      *tensorflow.Tensor
		wantError error
	}{
		{"dog image", "https://cdn.pixabay.com/photo/2019/07/30/05/53/dog-4372036__340.jpg",
			tensor1, nil},
	}
	for _, tt := range tests {
		testName := fmt.Sprintf(tt.name)
		t.Run(testName, func(t *testing.T) {
			response, e := http.Get("https://cdn.pixabay.com/photo/2019/07/30/05/53/dog-4372036__340.jpg")
			if e != nil {
				log.Fatalf("unable to get image from url: %v", e)
			}
			defer response.Body.Close()
			ans, err := imageToTensor(response.Body, m.createTensorMock, m.runSessionMock, tensorflow.NewGraph(), tensorflow.Output{}, tensorflow.Output{})
			if ans != tt.want || err != tt.wantError {
				t.Errorf("did not get the correct tensor")
			}
		})
	}

}

//Labels tests
func TestLabels(t *testing.T) {
	var labels []Label
	labels = append(labels, Label{"hello", 5.0})
	labels = append(labels, Label{"world", 4.0})
	labels = append(labels, Label{"this", 3.0})
	labels = append(labels, Label{"is", 2.0})
	labels = append(labels, Label{"a", 1.0})
	var labels2 []Label
	labels2 = append(labels2, Label{"hello", 5})
	labels2 = append(labels2, Label{"world", 5})
	labels2 = append(labels2, Label{"this", 3})
	labels2 = append(labels2, Label{"is", 2})
	labels2 = append(labels2, Label{"a", 1})
	var tests = []struct {
		name          string
		labels        []string
		probabilities []float32
		want          Labels
	}{
		{"normal labels", []string{"hello", "world", "this", "is", "a"}, []float32{5, 4, 3, 2, 1}, labels},
		{"top 5", []string{"hello", "world", "not shown", "this", "is", "a"}, []float32{5, 4, 0, 3, 2, 1}, labels},
		{"not in order originally", []string{"world", "hello", "this", "is", "a"}, []float32{4, 5, 3, 2, 1}, labels},
		{"tied labels", []string{"hello", "world", "this", "is", "a"}, []float32{5, 5, 3, 2, 1}, labels2},
	}
	for _, tt := range tests {
		testname := fmt.Sprintf(tt.name)
		t.Run(testname, func(t *testing.T) {
			ans := getTopFiveLabels(tt.labels, tt.probabilities)
			if !ans.Equals(tt.want) {
				t.Errorf("did not get the correct labels:\n got: %s \n want: %s \n", ans.getLabels(), tt.want.getLabels())
			}
		})
	}
}

func TestLabelsHelpers(t *testing.T) {
	var labels []Label
	labels = append(labels, Label{"hello", 5})
	labels = append(labels, Label{"world", 4})
	labels = append(labels, Label{"this", 3})
	labels = append(labels, Label{"is", 2})
	labels = append(labels, Label{"a", 1})
	labels2 := []Label{{"hello", 5}, {"world", 4}, {"this", 4},
		{"is", 5}, {"a", 5}, {"test", 5}}
	labels3 := []Label{{"hello", 4}, {"world", 5}, {"this", 4},
		{"is", 5}, {"a", 5}, {"test", 5}}
	labels4 := []Label{{"hello", 5}, {"world", 4}, {"this", 3},
		{"is", 2}, {"a", 1}}
	var tests = []struct {
		name       string
		labels     Labels
		wantLength int
		wantLess   bool
		wantEquals bool
		//wantGetLabels string
	}{
		{"normal labels", labels, 5, true, true},
		{"longer than 5 labels", labels2, 6, true, false},
		{"weird order", labels3, 6, false, false},
		{"normal labels same content", labels4, 5, true, true},
	}
	for _, tt := range tests {
		testname := fmt.Sprintf("%s test for length", tt.name)
		t.Run(testname, func(t *testing.T) {
			ans := tt.labels.Len()
			if ans != (tt.wantLength) {
				t.Errorf("did not get the correct length: got %d, want %d", ans, tt.wantLength)
			}
		})
		testname = fmt.Sprintf("%s test for swapping labels", tt.name)
		t.Run(testname, func(t *testing.T) {
			oldFirst := tt.labels[0]
			oldSecond := tt.labels[1]
			tt.labels.Swap(0, 1)
			if oldFirst != tt.labels[1] || oldSecond != tt.labels[0] {
				t.Errorf("Did not swap the labels correctly")
			}
		})
		testname = fmt.Sprintf("%s test for first element less than second", tt.name)
		t.Run(testname, func(t *testing.T) {
			ans := tt.labels.Less(1, 0)
			if ans != (tt.wantLess) {
				t.Errorf("did not get the less than correct")
			}
		})
		testname = fmt.Sprintf("%s test for two lists of labels being equal to orginal", tt.name)
		t.Run(testname, func(t *testing.T) {
			ans := tt.labels.Equals(labels)
			if ans != (tt.wantEquals) {
				t.Errorf("The lists of labels being equal should be %t but they are %t", tt.wantLess, ans)
			}
		})
	}

}

import os
import matplotlib.image as mimg
import matplotlib.pyplot as mplot
import cntk
from cntk.layers import Dense,default_options
import numpy as np

np.random.seed(0)

inp_dim = 784
output_classes = 10

def create_reader(path,training,inp_dim,output_classes) :
	label_stream = cntk.io.StreamDef(field="labels",shape = output_classes,is_sparse=False)
	feature_stream = cntk.io.StreamDef(field="features",shape = inp_dim,is_sparse=False)
	deserializer = cntk.io.CTFDeserializer(path,cntk.io.StreamDefs(labels=label_stream,features=feature_stream))
	batch = cntk.io.MinibatchSource(deserializer,randomize=training,max_sweeps=cntk.io.INFINITELY_REPEAT if training else 1)		
	return batch

data_found = False

test_file = os.path.join("data","Test.txt")
train_file = os.path.join("data","Train.txt")

if os.path.isfile(test_file) and os.path.isfile(train_file) :
	data = True

if not(data) :
	raise Exception("Train/Test file not found")


input = cntk.input_variable(inp_dim)
labels = cntk.input_variable(output_classes)

def net_model(feature) :
	with default_options(init = cntk.glorot_uniform()) :
		layers = Dense(output_classes,activation = None)(feature)
		return layers

input_layer = input/255.0
net = net_model(input_layer)

loss = cntk.cross_entropy_with_softmax(net,labels)
error = cntk.classification_error(net,labels)

learning_rate = 0.2
learning_schedule = cntk.learning_rate_schedule(learning_rate,cntk.UnitType.minibatch)
learner = cntk.sgd(net.parameters,learning_schedule)
trainer = cntk.Trainer(net,(loss,error),[learner])

def cumulative_avg(arr,diff=5) :
	if len(arr)<diff :
		return arr
	return [val if ids<diff else np.cumsum(arr,axis=None)/5 for ids,val in enumerate(arr)]

def print_progress(trainer,minibatch,freq,flag=True) :
	loss = float()
	error = float()

	if minibatch%freq == 0 :
		loss = trainer.previous_minibatch_loss_average
		error = trainer.previous_minibatch_evaluation_average
		if flag :
			print "MiniBatch:{} , LOss:{} , error:{} /n".format(minibatch,loss,error)

	return minibatch,loss,error

minibatch_size = 60
num_sweep = 10
sample_per_sweep = 600000
num_minibatch = (sample_per_sweep*num_sweep)/minibatch_size

reader = create_reader(train_file,True,inp_dim,output_classes)

input_map = {labels : reader.streams.labels ,
	     input : reader.streams.features ,
	    }

output_freq = 500
plot_data = {"miniBatch":[] , 
	     "loss" : [] ,
	     "error" : []
	    }


for iter in range(0,int(num_minibatch)) :
	data = reader.next_minibatch(minibatch_size,input_map = input_map)
	trainer.train_minibatch(data)
	minibatch,loss,error = print_progress(trainer,iter,output_freq)

	if not(loss == 0 or error == 0) :
		plot_data["miniBatch"].append(minibatch)
		plot_data["loss"].append(loss)
		plot_data["error"].append(error)


mplot.figure(1)
mplot.subplot(211)
mplot.plot(plot_data["miniBatch"],cumulative_avg(plot_data["loss"]),"b--")
mplot.xlabel("miniBatch")
mplot.ylabel("Loss")
mplot.title("minibatch vs loss")
mplot.show()

mplot.subplot(212)
mplot.plot(plot_data["miniBatch"],cumulative_avg(plot_data["error"]),"r--")
mplot.xlabel("miniBatch")
mplot.ylabel("error")
mplot.title("minibatch vs error")
mplot.show()


test_minibatch_size = 60
test_sample_per_sweep = 10000
num_minibatch = test_sample_per_sweep/test_minibatch_size
test_result = 0.0

reader = create_reader(test_file,False,inp_dim,output_classes)

test_input_map = {labels : reader.streams.labels ,
	     input : reader.streams.features ,
	    }

for iter in range(0,int(test_num_minibatch)) :
	data = reader.next_minibatch(test_minibatch_size,input_map = test_input_map)
	error = trainer.test_minibatch(data)
	test_result += error

print "Average test error {0:.3f}%".format(test_result)


out = cntk.softmax(net)

eval_reader = create_reader(test_file,False,inp_dim,output_classes)
eval_map = {input : eval_reader.streams.features}
eval_minibatch = 25

data = eval_reader.next_minibatch(eval_minibatch,input_map = eval_map)

img_label = data[labels].asarray()
img_feature = data[input].asarray()

img_eval = [out.eval(img_feature[i]) for i in range(len(img_feature))]

pred = [np.argmax(img_eval[i]) for i in range(len(img_eval))]
get_label = [np.argmax(img_label[i]) for i in range(len(img_label))]

print "Labels     : ",get_label[:25]
print "Prediction : ",pred












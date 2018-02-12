import gzip
import numpy as np
import struct
import os
import sys
import matplotlib.image as mimg
import matplotlib.pyplot as mplot
import shutil
from urllib  import urlretrieve


def loadData(src,cimg) :
	print "Download"+src
	zipfile,h = urlretrieve(src,"./delete.me")
	print "Download Complete"
	try :
		with gzip.open(zipfile) as zf :
			num = struct.unpack("I",zf.read(4))
			if num[0] != 0x3080000 :
				raise Exception("Magic number is not matching")

			num = struct.unpack(">I",zf.read(4))
			if num[0] != cimg :
				raise Exception("inavlid format")

			ccol = struct.unpack(">I",zf.read(4))[0]
			crow = struct.unpack(">I",zf.read(4))[0]
			if crow!=28 and ccol!=28 :
				raise Exception("28 rows and columns not there")

			res = np.fromstring(zf.read(cimg*ccol*crow),dtype=np.uint8)
	finally :
		os.remove(zipfile)

	result = res.reshape((cimg,crow*ccol))
	return result

def loadLabel(src,cimg) :
	print "Download"+src
	zipfile,h = urlretrieve(src,"./delete.me")
	print "Download Complete"
	try :
		with gzip.open(zipfile) as zf :
			num = struct.unpack("I",zf.read(4))
			if num[0] != 0x1080000 :
				raise Exception("Magic number is not matching")

			num = struct.unpack(">I",zf.read(4))
			if num[0] != cimg :
				raise Exception("inavlid format".format(cimg))

			res = np.fromstring(zf.read(cimg),dtype=np.uint8)
	finally :
		os.remove(zipfile)
	result = res.reshape((cimg,1))
	return result

def try_download(data_src,label_src,cimg) :
	data = loadData(data_src,cimg)
	label = loadLabel(label_src,cimg)
	result = np.hstack((data,label))
	return result



url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
num_train_samples = 60000

print("Downloading train data")
train = try_download(url_train_image, url_train_labels, num_train_samples)
print type(train)

url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
num_test_samples = 10000

print("Downloading test data")
test = try_download(url_test_image, url_test_labels, num_test_samples)


'''sample = 5001
mplot.imshow(train[sample,:-1].reshape(28,28), cmap="gray_r")
mplot.axis('off')
print("Image Label: ", train[sample,-1])'''

def savetxt(file,ndarray) :
	dir = os.path.dirname(file)
	
	if not os.path.exists(dir) :
		os.makedirs(dir)

	if not os.path.isfile(file) :
		with open(file,"w") as f :
			label = list(map(' '.join,np.eye(10,dtype=np.uint8).astype(str)))
			for row in ndarray :
				row_str = row.astype(str)
				label_str = label[row[-1]]
				feature_str = ' '.join(row_str[:-1])
				
				f.write('|labels {} |features {}\n'.format(label_str,feature_str))
	else :
		print "The file already exists"	




data_dir = os.path.join("data")

savetxt(os.path.join(data_dir,"Train-28x28_cntk_text.txt"),train)
savetxt(os.path.join(data_dir,"Test-28x28_cntk_text.txt"),test)
print "Done Bro"

## Week 4 (1/28/19 - 2/3/19)
# Weekly Goals:
* [ ] Build a hardware NN for *inference only*
* [ ] Run hardware NN on MNIST test dataset and record accuracy, latency, etc
* [ ] Have presentable results on first experiments and process by Friday, Feb 1

## Week 3 (1/21/19 - 1/27/19)
# Weekly Goals:
* [X] Have a trained, one-layer FFNN in PyTorch by the end of Friday, Jan 25
* [X] Plan out hardware NN- components, how they will link together
* [X] Plan out transfer between PyTorch and PyRTL- MNIST test dataset, trained weights

**Saturday, 1/26 (2.5 hrs)**
* Team meeting without Dylan, who is sick :( We ran the modified PyTorch NN for 5 epochs and got over 70% accuracy, which was surprisingly good.
* We planned out how we are going to tackle the transfer to PyRTL- since it's just one layer, we just need to perform one matrix multiplication between the input vector (each vector represents one image) and the matrix of weights, then apply the activation function to each element of the output. In the PyTorch NN, we used a ReLU as our activation function, which we have already composed in PyRTL. 
* We also worked on finding solutions to the two problems we could immediately forsee:
  * *Reading MNIST values into PyRTL*: We had discussed this as a group last week, but we weren't certain how we would actually give the hardware neural network the inputs, since MNIST is stored as images. Instead, we had considered generating and using a simplified version of a numerical database that represents integers as blocks of 1s and 0s. The discussion wasn't really settled last week, but this week we felt that we should focus on finding a way to read the MNIST testing dataset into PyRTL. Today, we learned that PyTorch has a built-in function that converts each image to a tensor when it sets up the train and tests sets. That means, if we can just access that same set in our PyRTL code, we can access and manipulate the MNIST entries just like matrices.
  * *Porting the trained weights into PyRTL*: Another we noticed with the PyTorch NN is that the initial weights are assigned somewhat randomly (based on a certain distribution- more details can be found [here](https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073)), so they will be trained an adjusted slightly differently everytime. That means we would have to re-assign the weights in the hardware NN every time as well. We experienced some trouble reading the full matrix of weights into an external file to read into the PyRTL file.
* The solution we are currently considering is to simply put both NNs into the same file, which would allow us to feed the weights directly into the PyRTL NN, written right below it. Eventually, we will modularize and streamline our process, but this seemed like the best way to simplify things to make sure the NNs themselves work, and get some tangible results. We'll add more moving parts such as writing/reading to an external file as we scale up.
* Two things we discussed but haven't found solutions for are: 
  * *Floating point number weights*: All of the weights we examined in the PyTorch NN are floating point numbers less than 1. We're not sure whether to scale them larger for the PyRTL NN, or to simply represent them with 1s and -1s, since we want to keep the initial hardware NN close to the software NN.
  * *nXn matrix multiplier*: We have a good, object-oriented matrix multiplier, but it can only multiply square matrices of the same dimensions together. Even with our simple model, we need to multiply a [10 X 784] matrix of weights by a [784 X 1] input vector. Our proposed solution is to just make two [784 X 784] matrices, padded out with 0s where the weights and input matrices/ vector would have no values. This will give us a [784 X 784] matrix, with our ultimate result held in the first 10 entries of the first column. While this will yield the correct result mathematically, it's going to be very inefficient, especially since matrix multiplication is an O(n^3) operation, and the test dataset has 10,000 elements. We may just go ahead and try it, but hopefully we can make some improvements.

**Friday, 1/25 (2.5 hrs)**
* Meeting with Deeksha- we were going about the problem all wrong, and it's actually a lot easier than we thought! We only actually have to represent the inference step in hardware- there will be NO training or backprop using PyRTL. All we have to do for now is build a super simple, working NN in PyTorch, starting with just one layer (no hidden layers), train it to at least 60% accuracy on the MNIST dataset, and then put the weights from the trained NN directly into the hardware NN and see the difference in behavior. At this point, we don't want to make any changes to adapt the software NN to behave more similarly to the hardware NN. Rather, we will try to approximate the simple, standard NN as closely as possible in hardware and see how far off its behavior is from what we would expect from the software NN. 
* Deeksha mentioned that we should figure out some sort of pipelining for our process, so we can quickly and easily shift the trained weights into the PyRTL NN after training the PyTorch NN to satisfaction.
* By next Friday, we need to have constructed and tested the PyRTL NN, with as few deviations from the PyTorch model, and have results from its inference to present. We have most of the pieces we need to put the hardware NN together (a matrix multiplier, MAC, and we'll need a working FIFO), so we should be able to have something running by then.
* Dawit and I spent met up later to finish up the PyTorch NN. Deeksha had recommended we start with one of the Pytorch NNs we had written last quarter by following tutorials, and just removing anything extraneous to reduce it to just a single-layer NN. The code I had written and modified last quarter was a CNN, but Dawit had written a purely feedforward NN, so we used that as a base. I took out the extra layers, adjusted the dimensions of the remaining layer to [28^2 by 10] (since the MNIST images are 28X28 pixel grayscale images, and we wanted to narrow the output to a 10-vector), and adjusted the calculations to account for the changes. Eventually, we had it running with 57% accuracy in 3 epochs. It ran really slowly, so we didn't test it on more epochs at the time.

**Tuesday, 1/22 (2 hrs)**
* Maggie and I met up to work on writing the loss function and backpropagation for the Pytorch NN, but we kept getting stuck trying to find a loss function that would yield integer numbers, since they all involve a division step. We can only represent division in powers of 2 with PyRTL, so we looking for something that we could port exactly to hardware. Spent a lot of time reviewing NN math, and looking at different possible loss functions and ways to adjust the weights directly in PyTorch if we can't use the built-in optimizer (which handles backprop), since we weren't sure if we could represent the same process in hardware. 

## Week 2 (1/14/19 - 1/20/19)

**Friday, 1/18 (2 hrs)**
* First meeting with Deeksha. She gave us a timeline of milestones for our project, and recommended that we start splitting up the workload among ourselves. Our main obstacle is that we will have to implement the software neural network using entirely integers, so that we can construct an equivalent NN in hardware without having to deal with floating point numbers in PyRTL.

* Meeting with Professor Mirza. Updated her on the status of our project and our progress. 

*Software milestones by Feb 1st:*
* Pick a problem: dataset, classification vs. regression. Build a simple PyTorch model (MNIST classification will be good for the final version). Train the network until you reach good accuracy (for MNIST ~ 95%)

*Hardware milestones by Feb 1st:*
Plan hardware -- what kind of modules do you need? If you don't have them, build them. How would you connect these? 

**Saturday, 1/19 (2.5 hrs)**
* Team meeting. Dawit shared a site he found that offered a way to build an integer-based NN, which we should be able to apply to make an activation function that approximated the distribution of a sigmoid using only whole numbers. We will have to develop our own loss functions and backpropagation that avoid floating points as well, however.
* Split up the work so that Maggie and I are working on making the loss and backprop functions with integer values, and Dylan and Dawit will work on figuring out how to access and update the weights, as well as incorporating the activation function Dawit found.

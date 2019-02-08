## Week 5 (2/4/19 - 2/10/19)
# Weekly Goals: 
* [ ] Read OpenTPU README on ArchLab github more closely
* [x] Access ground truths of MNIST images in test set
* [x] Determine which vectors are correct for each input
* [x] Speed up simulation and computation for PyRTL NN!
  * [x] Write a block matrix algorithm in numpy, figure out how to translate into PyRTL
* [ ] Measure accuracy of outputs from PyRTL NN
* [ ] Have presentable results on first experiments and process by Friday, Feb 8

**Thursday, 2/7 (6 hrs)**
* **Morning followup:** The program did not finish running, but it told me there was a syntax error after I hit ctrl-C to end it. I don't know why this didn't end the program.
* Fixed the syntax errors- I had forgotten to change a few variables that Dylan had declared that I had decided not to use (since I am not using the `getWeights` and `getImage` functions- everything is being done within the main for now). Turns out I had the dimensions of the matrices reversed, so I had to go back and adjust the dimensions and loops to fix that. 
* I'm just testing with a single image for now. If it works through the whole PyRTL NN, then it should be easy to wrap everything in a loop to iterate through the test dataset.
* I was able to get through inference for a whole input image, and we have an output vector now! There were a few PyRTL-related syntax errors before we could get to the simulation step. 
* Runtime is very slow- around 10 minutes for the whole program (until it hits an error) and that's just for one image. Even with just 100 images from the dataset, rather than 10,000, that's pretty slow. Much faster than with the monolithic matrix multiplier tho- we're finally getting through the "forward" function (all the multiplication/ relu)
* Finally got to simulation, and we are hitting a PyRTL internal error. Not sure how to resolve right now :(

**Wednesday, 2/6 (5 hrs)**
* Met with Dawit to create some plots for our PyTorch model. We set up a plot that charts Accuracy vs. Num Epochs and tested it with a loop of up to 5 epochs (data for 0, 1, ...5 epochs worth of training on the same model), and the chart is looking pretty good. Dawit is later going to set the loop to a much higher number and let it run all night, so we can hopefully achieve >80% accuracy and get a dope graph at the same time.
* Worked on block matrix algorithm as a possible solution to RAM problem. Wrote and tested the entire algorithm in `numpy` (pure Python, basically), then created a new branch of all our up-to-date code and reworked the PyRTL setup to incorporate my block matrix solution. It's running right now, but I'll know whether it worked or if I don't have enough RAM in the (later) morning. 
  
**Tuesday 2/5 (1 hr)**
* Met with Maggie, figured out how to access ground truths for each image in the 4-dimensional tensors
* While trying to figure out how to interpret the output vectors to compare them to the ground truth to determine accuracy, we met with Dylan. Turns out he had already written a function to do so- checks for the largest value in the 10-vector, and the index of that value is the network's guess for that image (each item in the vector corresponds to the "confidence" in that guess, so the most confident is taken as the final answer).
* We're not certain if this will actually work, since the inputs and weights are scaled so much, and we can't tell yet since we can't get the NN to run to completion


## Week 4 (1/28/19 - 2/3/19)
# Weekly Goals:
* [X] Build a hardware NN for *inference only*
* [ ] Run hardware NN on MNIST test dataset and record accuracy, latency, etc

**Saturday, 2/2 (2 hrs)**
* Met up briefly to discuss our goals and assign portions to group members
 * Maggie and I will be determining how to get the ground truths and measure the accuracy of the PyRTL NN
* Dylan ran the PyRTL NN later for a few hours, but later informed us that his laptop didn't have enough RAM to even get past the simulation step
 * Taking so long to simulate the network in hardware, the program didn't even get to the multiplication/ inference step
 * I suggested switching the implementation to block matrices to see if it would speed things up, but Dylan doesn't think that will speed up the actual simulation of the hardware
 * We also decided to only use the first 100 elements in the test dataset, rather than 10,00

**Friday, 2/1 (2 hrs)**
* My suspicion was correct- each data item = 4 images
* Dylan has started feeding in individual vectors into the NN hardware we currently have, but it is very slow. Inputs are scaled up by 256 (to shift floating points values into RGB form, as images are usually stored)
* Proposed block matrices as a possible solution to speed up matrix operations
 * A few benefits from using block matrices: Faster matrix multiplication, smaller area and power usage (since we can reuse a smaller matrix multiplier several times, rather than creating a single large multiplier in hardware
* Met with Deeksha- according to the milestones chart, we need PyRTL data by the end of *next* week
* Notes from meeting with Deeksha below:

   > Read ways to do quantization with neural networks to avoid losing precision/ accuracy
   > Make some graphs for PyTorch data- Accuracy vs. Num Epochs (should be decreasing)
   > Get accuracy up on PyTorch- run for about an hr of epochs- aim for at least high 80s in accuracy
   > Consider adding biases- Ax+b makes more accurate, b is the bias
       >Shifting up and down- can get biases in loss function, updated thru backpropagation
   > Weights/bias are trainable parameters, input is fixed
   > If pyrtl is too slow, break down into smaller block matrices 
   > Consider different ways of shifting data to all positive nums (renormalize)
   > PyRTL/python profiler?- tell us what program is doing while it is doing it- track memory leaks, loops, etc
    


**Thursday, 1/30 (4.5 hrs)**
* Read through an overview of the microarchitecture of the ArchLab's OpenTPU
* Spent a few hours reading pytorch documentation and tracing through the PyTorch NN code. Some observations:
 * The dataloader is being iterated thru only 2500 times, but the size of the dataset in the test dataloader is definitely 10,000
 * The batch size is 4- doesn't seem like a coincidence that the tensors are coming in sets of 4
 * I tried accessing each "dimension" of the tensors, and realized that each is a 28\*28 matrix with values between 0 and 1, exactly as expected for a single MNIST tensor
   * going off these observations, it would make sense for each "dimension" to represent a single MNIST image
 * Looking at the pytorch code behavior in the training and testing functions- it looks like the linear layer is taking in the full tensor, reshaped to a 4\*784 matrix, which it must be splitting up into 4 individual input vectors, since it requires vector inputs for the matrix math
    * Therefore, I'm pretty sure my idea was correct, and each data item in the dataloader holds 4 images' worth of information
 * Learned how to use .view() to shape each square matrix into a 784-vector
* Later, we had a full team meeting- still figuring out what to do with 4d tensors
 * Some disagreement over my idea- we read somewhere else that each dimension should be a color channel, so we weren't sure if the 4 28\*28 matrices all provided information about the same file (if we needed all 4 dimensions to describe a single image), or each corresponded to a different one and they were simply given in batches of 4
 
**Wednesday, 1/29 (2.5 hrs)**
* Met up with Maggie and Dylan- fed the trained weights into our PyRTL NN. We are scaling each weight by 1000 so we can work with whole numbers.
* We thought we would easily be able to access the tensors that the MNIST images were translated into, but the dataloader items gave us 4-dimensional tensors, which we hadn't been expecting. We couldn't determine what to do with each set of tensors, or why they were stored in 4 dimensions.

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
* Team meeting. Dawit shared [this](http://sharpneat.sourceforge.net/research/integer-neuralnet/integer-neuralnet.html) site he found that offered a way to build an integer-based NN, which we should be able to apply to make an activation function that approximated the distribution of a sigmoid using only whole numbers. We will have to develop our own loss functions and backpropagation that avoid floating points as well, however.
* Split up the work so that Maggie and I are working on making the loss and backprop functions with integer values, and Dylan and Dawit will work on figuring out how to access and update the weights, as well as incorporating the activation function Dawit found.

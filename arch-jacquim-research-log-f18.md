## Important Links:
[*My Reading Log*](https://docs.google.com/document/d/1w88Sf5nNbMch-iYMC7N0dAdIlIEadZRttGgzu_mLrHc/edit)
---
---
# Week 8 (11/15-11/21)
- [ ] Make edits on proposal after peer review comments
- [ ] Convert doc to LaTEX before we work on Week 9 draft!
- [ ] Read up on estimation functions in PyRTL
- [ ] More hardware work
- [x] Finish PyTorch MNIST example
  - [ ] Some extra stuff with PyTorch CNN (ongoing)
- [ ] Figure out how to get matplotlib to display from terminal

**Thursday, Nov 22 (3.5 hrs)**
* Happy Thanksgiving!
* Finished PyTorch MNIST NN tutorial, following [this](https://nextjournal.com/gkoehler/pytorch-mnist) demo. This tutorial shows how to build and train a convolutional neural network for image recognition. The neural network itself seems to work fine, and I noticed definite improvements in accuracy over 3 epochs:
  * Epoch 0 (no training, randomly initialized parameters):
    * Accuracy: 3% to 10% over several runs, Avg Loss: ~2.3
    * The following results are from just 1 run, since it takes a while to complete all three epochs
  * Epoch 1:
    * Accuracy: 94%, Avg Loss: 0.1944
  * Epoch 2: 
    * Accuracy: 96%, Avg Loss: 0.1302
  * Epoch 3:
    * Accuracy: 96%, Avg Loss: 0.0977
 * *Problems*: 
   * I had a few minor problems with syntax, and had some problems saving the state of the model each iteration of the train 
loop (It kept saying that the directory didn't exist, and I eventually figured out that I needed to use 'results/model.pth' rather than '/results.model.pth'- same for the optimizer state_dict() saves. I also had to create the 'results' directory by hand.)
   * My main problem is that matplotlib doesn't seem to be working from my terminal. The demo I followed goes on to use matlplotlib for a bunch of other tests, and for viewing some of the MNIST entries before constructing the NN. However, when I used matplotlib, I got *no* output, but no error either. The graphs simply don't display from my terminal. This wasn't a huge deal here, since the printed output to my terminal was enough to see the progress being made by my NN, but I can see how matplotlib would be a tremendous help in the future for graphing and analyzing the output data. I've heard that this is a pretty common problem, so hopefully I can figure it out and get plotting soon.
* I also started thinking about how we're going to translate PyTorch NN code to PyRTL. I noticed that PyTorch just has pre-built Linear layers in torch.nn, which are used as the fully-connected layers in the CNN that I built for the MNIST example. Going off what Deeksha told us last Friday, I would guess that these Linear layers are essentially feedforward layers/NNs?
  * Does this mean that we would simply have to figure out how to implement a PyTorch nn.Linear layer in PyRTL for the first part of our experiment. Maybe we can find the implementation for a Linear layer and go about translating it into hardware?
  
* I decided to mess around with the CNN that I built. Here's some stuff I did:

  1. Added more training epochs:
    * I wanted to see if more generations = more accuracy, since I noticed that the accuracy seemed to be capping out at 96% with just 3 epochs, so I increased the number of epochs to 10.
      * After it completed 10 epochs, the final accuracy was just 98%. Obviously, this is pretty high, but not such a huge improvement given that I ran it for over 3 times as many epochs. Going back through the output, I noticed that the NN actually hit 98% in its 8th epoch, and was at 97% from its 4th to 7th epochs. So the rate of increase of accuracy is *very* slow. 
      * Speaking of slow, the program itself took a while to complete. The output printed along the way, so I could track the NN's progress in real time, but it took several minutes for execution to finish. I didn't actually time it (I had it running in the background), so maybe the next thing I want to do is track the execution time and see how it grows as I add more epochs.
      * It would be nice to have matplotlib working when I do that, but there is probably as way to do it with just outputs to terminal
      
  2. Allowed number of epochs to be set by the user before building and training the CNN:
     * Changed declaration of `n_epochs` to `n_epochs = int(input("How many epochs? "))`
 
  3. Tracked runtime:
     * Used the `time python3` command to run my program, which runs it as normal and then prints out the real, user, and system runtimes. 
       * 1 Epoch (94% accuracy):
         * `real`: 0m 35.009s
         * `user`: 1m 2.063s
         * `sys`: 0m 8.203s
       * 3 Epochs (96% accuracy):
         * `real`: 1m 33.407s
         * `user`: 3m 12.297s
         * `sys`: 0m 17.453s
       * 10 Epochs (98% accuracy):
         * `real`: 4m 58.418s
         * `user`: 10m 26.578s
         * `sys`: 0m 53.500s
       * 15 Epochs (98%):
         * `real`: 7m 5.096s
         * `user`: 15m 16.344s
         * `sys`: 1m 16.172s
     * According to [this](https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1) post on StackOverflow, `real` = actual elapsed time, `user` = CPU time outside kernel and inside process, and 'sys' = CPU time in system calls within kernel. From what I understand, the user/sys times can add up to be greater than the real elapsed time with a multicore processor (since processes are split up and execute simultaneously, and there is some additional overhead).
       * I think my program is single-threaded, so is there some process happening so that I use more than one core in execution? I'll look into that if I have time, but it's not a huge question.
     * Clearly more epochs = more time, at a signficant rate of increase and almost no improvement in accuracy. I might be able to increase accuracy with larger datasets or more layers, so I might look into that next. 

* This is a more complicated example than what we're ultimately going to build in our project, so I should probably look more into building a standard feedforward NN in PyTorch and find a good sample dataset to run it with (since that's what we'll be dealing with when we build it in PyRTL)
  * **What kind of processes are FFNNs used for? A CNN is good for image recognition, so MNIST was a good fit, but I need to know what a standalone FFNN can do in order to choose the right training/testing datasets.**
         
**Monday, Nov 19 (1 hr)**
* Received peer feedback, a lot of edits planned for the proposal- The main points were:
  * More definitions of NN vocab- thinking about an appendix section
  * Be more clear about difference between NNs and NN accelerators
  * More clear problem statement

**Sunday, Nov 18 (2 hrs)**
* Read HCI proposal draft for peer review, made a lot of notes and filled out handout 
**Friday, Nov 16 (1.5 hrs)**
* Meeting with Mai:
 * Discussed our proposal and the new edits
 * Mai also mentioned some interesting topics that we might want to expand on in our project:
 > Overfitting vs underfitting in hardware (consume more energy? area?)
 > Are there papers on feedforward neural networks in hardware?
 > If FFNNs are not that common/important should we focus on another type of NN?
 > Or just use FFNNs as an introduction
 > GRUs LTSMs could have more impact? Common in machine learning
   > Single cells could be easy to implement and test
 > Maybe we also want to look at the effects of overfitting and underfitting in hardware?
 * As we understood, we're implementing a feedforward neural network specifically because it is simple, but Mai's discussion led us to question whether FFNNs are actually important/significant to NN study as a whole
   * We decided to take this question to Deeksha in our 11:00 meeting
   
* Meeting with Deeksha:
 * Quick discussion of the paper again, some extra advice from Deeksha and Professor Sherwood
 * Miscellaneous important things for hardware design:
   * always write tests
   * send specific inputs, compare output to expected values
   * when trying to use PyRTL, search documentation for specific helper functs
 * Re: Overfitting, underfitting hardware NNs
   * What are some practical uses for wanting to over/underfit data?
     * I mentioned that overfitting could be useful if we knew we would always be feeding a NN data that is very similar to the training data
     * For underfitting, maybe we don't need the NN to be as accurate in some operations
     * If we can show there are performance benefits to over/underfitting, then maybe we can recommend times where it is appropriate to over/underfit a NN
   * Obviously, this is sort of an "extra" question, but it was interesting, and Deeksha had us think of what we would need to know before we could start to think about the over/underfitting question
     * How will we determine over/underfitting? Will we just observe it incidentally as we manipulate other hyperparameters
     * Is it worthwhile/ useful for our proj? What questions do we need to answer before looking at over/underfitting?
* **Why FFNNs?**
    * **Feedforward are one of the simplest NNs, and they are actually used as the fully connected layers from CNNs and RNNs**
    * **By implementing and analyzing a FFNN, we get info about a key component of several other, more complex NNs (we have a foundation to generalize to other NNs**   
 * Things to do:
   * Build hardware neural networks
   * Which hyperparameters to adjust/how 
   * How to track results 
   * Now that we have a general idea of how to use PyTorch for NN design, and PyRTL for NN design, we need to obtain skills for the third part of the project
     * start estimation stuff- getting preliminary results- get familiar with testing
     * working on simple circuits- find energy, area, etc of what we have built (matrix multiplier, etc)
  * Re: the paper: 
     * Prof Sherwood recommended more clarity on why we care about NN accelerators
       * More power discussion- comments on Slack
       * Energy = power integrated over time
       * Power as thermal load is big concern in all hardware
        * Think about the intended audience for this proposal 
* We are being added to a repo on Overleaf so we can collab on LaTEX doc to format our proposal

# Week 7 (11/8-11/14)
- [x] Further reading on different architectures
- [x] More detailed reading of essays/posts sent by Deeksha
- [ ] Finish matrix multiplier
- [ ] Learn about FIFO, implement in PyRTL (optional)
- [x] Part 3 of paper due by class Wednesday (with edits on first parts): Evaluation and Implementation Plan
- [x] Create a timeline for work over next two quarters (Rough idea by class on Wednesday, try to add as much detail as we know about)

**Wednesday, Nov 14 (2 hrs)**
* Team meeting! We went over the additions that Maggie and I had made last night, as well as a bunch of comments that Deeksha had left on the most recent iteration of the paper (including the new paragraphs).

**Tuesday, Nov 13 (3.5 hrs)**
* Worked with Maggie to add two more paragraphs to our intro/related works section, according to Mai's suggestion. We read through our compiled summaries of the literature papers to identify the ones that seemed the most closely related to our research question and wrote a paragraph for each about the neural network hyperparameters discussed and the some specific arhcitectures designed to optimize their behavior. The papers we chose are: ["DNPU: An Energy-Efficient Deep-Learning Processor with Heterogeneous Multi-Core Architecture"](https://ieeexplore.ieee.org/document/8474942) and ["An Energy-Efficient Architecture for Binary Weight Convolutional Neural Networks"](https://ieeexplore.ieee.org/document/8103902)

**Monday, Nov 12 (2 hrs)**
* Team meeting instead of class, we worked on making edits to the intro and related works of our paper. Added a lot of citations and cleaned up the language to be more formal and explicit about the connections between the related works and our own research question. Resolved all of Mai's advised changes (hopefully) except being more clear about the PyTorch/PyRTL connection in our proposed solution section. 

**Friday, Nov 9 (4 hrs)**
* Meeting with Mai:
  * Mai had read over our first draft and gave us a lot of good advice on revisions. Photos of her notes on the paper are available on our shared Google Drive folder, accessible from our group GitHub page and [here] (https://drive.google.com/drive/folders/1TNXCO44omkWR_bspbvrR-L-IRbbHNyIT?ths=true)

* Meeting with Deeksha and Professor Sherwood:
  * Deeksha explained PyTorch step in more detail: 
    * Implement and test the neural network in PyTorch first to make sure it works, adjust hyperparameters to see effects in normal implementation
    * Then translate by hand into PyRTL to see behavior in hardware
    * Ideally, would be able to translate PyTorch to PyRTL automatically by end of the project, but that's just an extra goal
  * She told us to make our own ERSP folder on the ArchLab GitHub to push our hardware designs to as we work on them
  * After that, we just reviewed some of Mai's comments on our paper to determine what specific parts/papers to target

* Attended a talk with ArchLab: "Towards Hardware Security", with Professor Houman Homayoun
 * Not really related to our project, but the talk was pretty interesting and I might want to read more about security when I have time 

**Thursday, Nov 8 (2 hrs)**
* Team reorganized our literary search- reviewed intros and conclusions, put them into three main categories based on topic, then identified the neural network hyperparameters and tradeoffs discussed in each paper.
* Categories we divided the papers into:
  * NNs in hardware
  * Low power, high efficiency
  * Different architectures
* I also built the Python neural network, following the tutorial shared by Deeksha
  * Pure Python, no PyTorch or PyRTL
* Prepared presentation slide on week's activities for meeting with Deeksha

# Week 6 (11/1-11/7)
- [x] Proposal Pt 2 draft by WED 11/7: Proposed Solution
- [x] Meetings and goals with Mai, Deeksha, and ArchLab
- [x] Read essays and blog post sent by Deeksha
- [ ] Learn more about different architectures
- [x] Follow along with NN tutorial in Python- learn some more Python syntax
- [ ] Build a matrix multiplier in PyRTL, build a FIFO if have time
- [ ] Read more PyRTL documentation!

*Not a very productive research week, due to three back-to-back midterms*

**Wednesday, Nov 7 (2 hrs)**
* Team meeting:
 * We worked on refining our proposal: Wrote the proposed solution and edited our introduction according to Deeksha's advice

**Friday, Nov 2 (4 hrs)**
* Meeting with Mai at 8am: 
  * Mai asked a lot of specific questions about the physical implementation of our experiment, and we compiled a list of the ones we wanted to go over with Deeksha:
  > How are we going to use PyTorch with PyRTL? What role will each toolset play in our tests? What specific tradeoffs are we looking at?
  She also emphasized that we need to organize our literary search more carefully into clusters based on topic, and identify the most important papers for our project by next Friday. We also need to start looking into tutorials on full feedforward NNs, know how NNs scale into different layers and how they work with specific architectures, and know how the circuits we have built (MAC, ReLU) are relevant to the project.
  
* Meeting with Deeksha at 11am:
  * Meeting inputs:
    * Present slides for meeting about work done during week, plans for next week
    * Questions from meeting with Mai
    * What is PyRTL ultimately giving us as a final measure? What kind of data/output from tests should we be looking at?
    * How can we implement negatives and floating point numbers in PyRTL? (for use as weights, floats -1 through 1)
    * Review comments on proposal, see where we need to add more detail
  * Meeting outputs:
    * We will be incorporating PyTorch and PyRTL as part our research question
      * Write the NNs in PyTorch, figure out how to put into PyRTL
    * Negative numbers in PyRTL must be expressed with two's complement, but there are no floating point numbers, so we'll have to scale weights to integer values
    * Suggested Clusters for papers
      > 1) Applications of NNs- deep learning? (Implementations)
      > 2) NNs in hardware (Optimizations) 
      > 3) Low power, high efficiency in hardware
      > 4) Different architectures- TPU and 2 Microsoft Research papers (SIMD)
    * Discussed some different architectures:
      * RISC vs CISC architectures- tradeoff between complexity & amount of instructions
      * Google TPU- time dependent systolic array architecture
      * SIMD- single instruction multiple data- 1 instruction happens over and over on diff data (pipeline) 
         * More complex architecture does simple things faster
         * Microsoft paper uses SIMD
    * TPU etc are implementations of NNs in hardware, not the NN themselves- NN accelerators
      * Chips that behave like NNs- designed for NNs to run on them, to accelerate their performance 
    * How to compute neural networks in computer architecture? Examine effects of specific hyperparameters on performance
      * **Neural Network hyperparameters- num layers, activation function, learning rate, dropout, batch size**
    * Discussed more details of steps of PyRTL process- design (code the instructions to build the hardware), simulate (run it as though it were a real piece of hardware), synthesis (putting on chip- uses netlists/bit streams- synthesize hardware by putting it on a physical FPGA)
    * With regards to testing effects of hyperparamters, measuring energy consumption, area, etc, PyRTL comes with toolchains that estimate area based on feature size (transistor size, getting smaller) and determine other effects
      * Power/ Latency/ Timing estimates can be done w these toolchains as well
    * Learned a little about technodes (transistor size), how area is based on technode chosen, and how the transistor is the component whose size is being limited by quantum effects (as mentioned by Professor Sherwood before)
    * Our team discussed the proposed solution section of our paper, then showed it to Deeksha to see if we need to add more detail. So far, we have: Build a feedforward NN in PyRTL, adjust hyperparameters individually and test performance with toolchains to determine how specific hyperparameters affect various aspects of performance.
    
* Group work in ArchLab at 4pm:
 * Met up in ArchLab again to go over the blog post Deesksha sent us about building an NN in Python (not using PyRTL). I didn't want to follow along without understanding anything by just copy and pasting the code, so I spent a lot of time reading about specific syntax for Python and the specific libraries being used, as well as the math being discussed (learning rate, gradients, backpropagation, etc). I think I get the main idea pretty well, but the unfamiliar syntax is what is slowing me down the most

**Thursday, Nov 1 (1/5 hrs)**
* Looked over Deeksha's feedback on proposal: Definitely some terms that need to be cleared up or explained more thoroughly, and we'll definitely be making some refinements as we read the related works in more details. 
* Put together slide for meeting with Deeksha about weekly accomplishments, plans for the week
  * Mainly want to read more PyRTL documentation, so I can implement more complicated circuits with less help
  * More on implementation of NNs
  
# Week 5 (10/25-10/31)
**Goals:**
- [x]  Refine literature search down to 10 papers by MON- can Slack Deeksha if need advice on a particular paper
- [ ]  Make a simple NN with team, MNIST dataset by **FRI 11/2** (more PyRTL!) [Got new reading material this week on this topic]
- [x]  Make a MAC (multiply-accumulate) and ReLU (max(0,x) activation function) in hardware with PyRTL, assigned by Deeksha
- [x]  Read new material (either NN blog post sent by Deeksha, or book recommended by Mai)
- [x]  Get started on writing proposal with team- Research context/problem statement by **WED 10/31** (Start compiling related work?)
- [ ]  Read survey paper "Artificial NNs in Hardware..."
- [ ]  If time, continue video series on PyTorch recommended by Dawit

**Wednesday, Oct 31 (2 hrs)**
* Team meeting today, which was very productive.
  * Started by finishing up the parts of the proposal that were due by class time- The intro, related works, and problem statement. Deeksha did respond to our messages on Slack, saying that the sentence we sent her as our problem statement captures what we want to achieve with the project, it doesn't provide a context for the work. She said she could write a few sentences to set up the problem statement, but I wasn't sure she should just be directly telling us, so I went ahead and shared our draft of the proposal introduction to show her what kind of context we intended for the problem statement. We only got her message after our meeting ended however, so we couldn't get incorporate any of her advice while we were working. Hopefully she'll be able to help us refine the proposal.
  * Built and simulated the MAC and ReLU circuits in PyRTL! Dylan had read the documentation during the week, and is already familiar with circuit design, so he led us through the process of sketching out the circuits on paper, then writing and testing the circuits using PyRTL. We figured out what libraries and functions to include, and how to simulate the circuits with random inputs. Something we need to figure out is how to represent negative and floating-point numbers in PyRTL, since we'll be needing both of those for the weights in NNs (which take on floating-point values between -1 and 1). 

Circuit Diagram of a MAC Unit:

  ![alt text](https://www.researchgate.net/profile/Mathan_Natarajamoorthy/publication/283521704/figure/fig5/AS:412039555108871@1475249294009/Figure-5-Architecture-of-proposed-MAC-unit.png "MAC unit diagram")

Graph of Input vs Output of a ReLU Activation Function:

   ![alt text](https://i.stack.imgur.com/by4EB.png "ReLU graph")
   
Returns 0 if initial input x <=0, otherwise it just returns the inital input x.

Things to do soon:
  
*Look at NN implementations with PyTorch, using MNIST dataset*

*Read more of the three selected papers from the literature search*

**Monday, Oct 29 (2 hrs)**
* Started work on proposal: introduction, related works, problem statement need to be sketched out by next class (Wed). Our proposed problem statement so far is "How does altering specific hyperparameters of hardware feed forward neural networks affect energy consumption, accuracy, and area?" I messaged Deeksha on Slack to ask her whether this seemed to be hitting all the points, but she just asked more about the proposal itself and has not yet given feedback on our problem statement. I also started reading the abstracts and intros of the three papers we selected, since I wasn't familiar with some of them (chosen by my teammates). The survey paper, "Artificial NNs in Hardware...", looks like it has some pretty interesting/promising information, so I definitely want to read that more closely first to get a broader view of the field of our project.

**Sunday, Oct 28 (2 hrs)**
* Worked on looking through papers listed on literature review, read through abstracts and intros to narrow down our list. Ultimately, we selected three that we thought were most relevant: ["Artificial neural networks in hardware: A survey of two decades of progress"](https://www.sciencedirect.com/science/article/pii/S092523121000216X), ["Low-power, high-performance analog neural branch prediction"](https://dl.acm.org/citation.cfm?id=1521824), and ["An Energy-Efficient Architecture for Binary Weight Convolutional Neural Networks"](https://ieeexplore.ieee.org/document/8103902). Hopefully this will be enough to get started on the proposal.

**Friday, Oct 26 (2.5 hrs)**
Three meetings today! 

* Meeting with Mai:
  * Mai asked us a bunch of questions about what we know about the project and what specific knowledge we have about the topics so far. We talked about our reading on neural networks (perceptrons, gradient descent, learning rates, weights/matrix math, etc) and hardware/energy (parallelization, frequency scaling). 
  * She taught us about the Train/Dev/Test process of training neural networks and explained some specific terms about NNs:
    * Hyperparameter tuning: adjusting different variables withing the NN to make it more accurate
    * Overfit: When the NN is too closely tuned to the test data, and cannot generalize well to all data input after training
    * Train/Dev/Test data should be in a ratio of 60%/20%/20%
    * Also recommended we get and read the book "Make Your Own Neural Network: An In-Depth Visual Guide for Beginners"
  * Recommended project for the next few days: build a NN, get PyTorch running with MNIST dataset (handwritten numbers)

* Meeting with Deeksha, Professor Sherwood: 
  * Meeting inputs:
    * What should we focus our attention on/ What is our project going to involve? (Need to start narrowing scope)
    * Should we be focusin more on PyRTL or PyTorch?
    * Mai and Professor Sherwood mentioned hyperparameter tuning- what exactly are we doing with this?
    * How will two main topics (NNs, energy efficiency) come together specifically for our project? 
  * Meeting outputs:
    * Should look at papers centered on computer architecture with NNs (architecture implementing NNs), power (specifically energy efficient implementations of NNs)
      * What makes these things energy efficient? What is/isn't efficient, what cannot be improved
    * Look for papers from ICSA, ASPLOS, MICRO (conference and IEEE magazine), HPCA
      * Read abstract and intro in detail for literature search, briefly skim results section
      * Also might want to consider number of citations (more citations = more popular work), and specifically *hardware* implementations, not algorithms
    * From now on, make *weekly* slides about research activities for discussion during meetings 
      * Try to include at least one graphic
    * Focus on PyRTL, set up a NN soon
      * PyTorch more "optional" for now- still have to work on it
    * Asked Professor Sherwood about some of the PyRTL examples I had gone through, and he talked about the details of building one-bit and ripple-carry adders in hardware using PyRTL, waveforms for visualizing inputs and outputs of the adders

* ArchLab PyRTL Tutorial:
  * A lot of similar material to what Professor Sherwood talked about during our 11:00 meeting
  * More explanation of the simulation step of writing hardware in PyRTL
    * Simulation "builds" the hardware described by the "recipe" in the PyRTL file
  * Mentioned that it is getting harder to build faster processors as we have before (speed doubling at a much slower rate than before), and emphasized that there are other ways to build good/efficient processors
    * Redefine what is considered an improvement: People have been focusing on processor speed for so long because it is what is most immediately enticing/ obvious, but other aspects have been neglected-- ex, energy efficiency! 
    * Now what used to be "fringe work" with computer architecture (improving aspects other than speed, like energy consumption, etc) is becoming more normal, since researchers are reaching physical limitations to speeding up/ shrinking processors (quantum stuff)
    
 **Overall:** Got a better idea of direction of project and literature search. Need to focus on PyRTL, make a NN by next Friday. Deeksha will be sending us some more information on that topic soon.


# Week 4 (10/18-10/24)
**Goals:**
- [x] Finish reading "Power..." paper, take notes for pass 1 by Friday
- [x] Complete slide for ERSP presentation for Friday's meeting with Prof Sherwood and Deeksha
- [x] Go through PyRTL examples by Friday meeting
- [x] Teach topic during class time (parallel computing)
- [x] Download, look at PyTorch
- [x] More PyRTL if time, but definitely attend tutorial Friday 10/26
- [x] Read new paper when we get it
- [x] Finish compiling notes on "Power..."
- [x] Figure out how to link to specific entries in reading log
- [ ] More reading on neural networks- definitely want to look more at learning rate determination

**Wednesday, Oct 24 (2.5 hrs)**
* Team meeting! Dylan couldn't attend, so it was just me, Maggie, and Dawit. We focused on learning more about PyTorch by following along with a tutorial on the website. Learned what a tensor is, how to implement one in PyTorch, and some of the functions that come with them. Dawit recommended [this](https://www.youtube.com/playlist?list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m) video series on YouTube to learn more about PyTorch, so started watching the first few videos. By our meeting on Friday, I want to have at least built the predictor example from this video series (in which the program uses tensors to predict the outcome of a test based on the number of hours spent studying), which will help me learn more about the training process.
* Figured out how to link to specific places in my reading log, so went back and added links for each entry. Everything is still within the same document, however.

**Tuesday, Oct 23 (3 hrs)**
* Downloaded PyTorch, looked at online tutorial and made some tensor matrices
   * Tomorrow, during team meeting, we will be working on PyTorch together and hopefully be able to put together a demo by Friday
* Finished "In-Datacenter Performance..." paper, notes found [here](https://docs.google.com/document/d/1w88Sf5nNbMch-iYMC7N0dAdIlIEadZRttGgzu_mLrHc/edit#bookmark=id.9g3pjsqusfbf) 
   * Main takeaway:
   > The TPU co-processor (written using TensorFlow) was able to improve inference and response time of spoken commands significantly over traditional CPU and GPU servers. Its simple design used less energy than either traditional implementation (though its consumption did not scale well when operating at less than maximum load), reduced dependence on the CPU, and overall demonstrated that neural networks offered big improvements in inference applications. 
* Worked on literature search- Compiled 7-8 interesting/related research papers from IEEE and ACM databases. Links to the papers, as well as some basic information about each (author, year, abstract summary) are listed on a doc shared with my group [here](https://docs.google.com/document/d/1ASOt_Sq22KKWExKKTC8lCLQ7iYaSatrMa1QnUe9ziOI/edit). My contributions are indicated in purple.

**Monday, Oct 22 (1.5 hrs)**
* Started reading, taking notes on new paper "In-Datacenter Performance Analysis of a Tensor Processing Unit"
* Compiled rest of notes on "Power" paper, notes [here](https://docs.google.com/document/d/1w88Sf5nNbMch-iYMC7N0dAdIlIEadZRttGgzu_mLrHc/edit#bookmark=id.2msvdre2aznp)

~~*Start literature search- 10 potential papers by class on WED*~~

~~*Narrow down list of research questions during team meeting on WED*~~

**Sunday, Oct 21 (1.5hrs)**

Research Problem, Personal Brainstorm:
* Frequency scaling can speed up processors, but with a corresponding linear increase in required energy 
* Neural networks can improve branch prediction, and therefore speed up execution time
* Energy efficiency is a growing problem as more technology is built, esp with small items where the hardware components are at risk of being damaged by energy leakage
* Reducing supply voltage to processors cuts consumption the most, but also reduces their performance
* We need to find a way to reduce processor power consumption without affecting performance
* How do we design neural networks that run on minimal power? 

*What do you like most about the work you have been doing?  What do you like least?  What is the biggest concern or question you have?*

So far we’ve mostly just been doing reading and getting familiar with the Python-based tools we’ll be using for our actual work, but I think things are going pretty well. I’ve found all three weekly meetings (with the lab, team, and Professor Sherwood/ Deeksha) very helpful and interesting. With my teammates, it’s nice to discuss the papers, figure out PyRTL, and learn things from each other. We’re exposed to a lot of new information at once in the larger meetings, which is a little intimidating, but the other members of ArchLab are usually willing to answer all of our questions and explain topics we’re not familiar with. There’s nothing I really dislike so far, I’m mostly just worried about spending enough time on the project. I also have to learn Python while I use PyRTL and PyTorch, but I think I’ll be able to keep up.


**Friday, Oct 19 (1 hr)**
* Meeting with Deeksha, Professor Sherwood not present. Minutes below:
    * Meeting inputs
      * What exactly are wires/ WireVectors in PyRTL? 
      * Explanation of 'sim' step in all the PyRTL examples?
      * Prepared presentation slide on weekly progress for Deeksha
    * Meeting outcomes
      * Learned that 'wires' are the actual physical wires in the hardware, and about how the different lengths of the wires affect transfer of data/ execution of programs. Each wire can be thought of as a single bit of information, so a WireVector is a collection of bits associated with a particular piece of information. 
      * Some discussion on the importance of energy efficiency in embedded appliances as part of Internet of Things (IoT)- when the processor components are not needed, they shouldn't be active. Deeksha explained the concept of dynamic voltage frequency scaling (DVFS), a technique where voltage to a component is reduced or increasing depending on the circumstances, rather than just receiving a constant flow. 
      * Recommended to look at PyTorch, for neural networks using Python, maybe put together a demo for next meeting? 
      * There will be a PyRTL tutorial next Friday at 4:00pm
    * Things I thought of after the fact:
      * Maybe a little more clarification on hardware blocks and FPGAs, but I can probably find theses things out on my own. 
      * render_trace() in PyRTL?

**Thursday, Oct 18 (3.5hrs)**
* Finally worked through some PyRTL examples! I typed out the combologic and counter examples to follow along, and started the statemachine example as well. Definitely some questions, since I've never done hardware design/simulation before, but I felt surprisingly comfortable with the Python syntax. 
* Finished the "Power..." paper. It's not really a research paper, more kind of an article, but it made a lot of interesting points about parallel processing, voltage reduction, and the increase of leakage with miniaturization of technology. Notes are available on my reading log, but I haven't fully finished transcribing them yet (and will probably use a slightly different format, since the research paper questions are generally not applicable)
   * I definitely want to follow up on some of the topics mentioned in the paper (published in 2001), to see how the predictions made in the paper turned out
* Main takeaway from this paper:
> Processors are generally designed to have high performance, meaning faster execution of given instructions, often at a higher energy cost. Techniques like pipelining to improve performance rely on frequency scaling to speed up execution, but energy consupmtion increases linearly with speed. The paper discusses the three factors of energy consumption (supply voltage V^2, short-circuit current, and leakage) and ways to reduce these factors. Parallel processing is proposed as a way to reduce supply voltage without sacrificing performance. Leakage is a factor that is rarely addressed but becomes significant as technology shrinks, to the point where leakage could become the main source of power drain and could damage hardware component of microchips (thermal runaway).

# Week 3 (10/11-10/17)
**Goals:**
- [x] First meeting with Professor Sherwood this Friday- ask about lab hours
- [x] Set up Ubuntu on windows to use terminal for Python (wazzow!)
- [x] At least download PyRTL tool, hopefully start working with it
- [x] Make another pass over prev paper ("Dynamic Branch Prediction...")
- [x] Start new paper when Deeksha/ Professor Sherwood sends it to us

**Wednesday, Oct 17 (3.5hrs)**
* Met with team today for two hours. We spent part of the meeting talking about our teaching topics for class, and the other part setting up the PyRTL tool
    * Maggie and I had the extra step of downloading an Ubuntu distribution for Windows 10 so we could use the bash terminal, which took a while
    * Eventually got both Ubuntu and PyRTL up and running, downloaded vim as well to have a familiar text editor. I'm planning to go through some of the PyRTL examples on the lab GitHub, but I spent about half an hour trying to get familiar with using Python in the new environment. I'm not really familiar with Python programming, so I did a Hello World just to figure out some basic syntax and compilation commands. (I'm just using **python "file.py"** to compile and execute, but I'm not entirely certain that's right.)
    * Continued reading and taking notes on "Power...". It's pretty short, but there's a lot of interesting stuff being mentioned. The topic of parallel computing seems really intriguing, and I'm hoping it's related to the project that Professor Sherwood/ Deeksha have in mind because it seems like there's so much to learn about it/ things you can do with it. Of course, this paper is kind of old (2001, just like the previous paper), so there's probably plenty of new research for me to catch up on.
 

**Tuesday, Oct 16 (3hrs)**
* Started reading new research paper Deeksha sent to our team- "Power, Architectural Design Constraint..."
* Chose parallel processing as my teaching topic, and put together a [lesson presentation](https://docs.google.com/presentation/d/1yCxJnPQI76wYQ2jj_bZSJ5TVwo5UWme4EmkmRED93lE/edit?usp=sharing)
    * Main topics: Defining parallel computing/ processing, talking about Amdahl's Law, applications to neural network training and energy efficiency
* Meeting with entire ArchLab today at 1:00pm. Mostly just for everyone to introduce ourselves to each other, but we ended up talking about PyRTL a lot since basically everyone in the lab is using it for different projects. Professor Sherwood mentioned new updates had just been added that made it possible to use PyRTL on the Jupyter Notebook Python IDE, but Maggie and I still should figure out how to set up Ubuntu to use the terminal on Windows 10. The master's/ PhD students talked about some pretty specific parts/applications of PyRTL, and so I asked a couple questions in order to try to keep up. I know have a better understanding of clocks/ synchronization of clock domains. 
* We are also planning a weekly lab-wide meeting in addition to our other meetings, so a poll has been sent out to find the best time for everyone to meet up.

**Sunday, Oct 14 (2hrs)**
* Read second pass on "Dynamic Branch Prediction....", filled out questions for part 2 of reading assignment. Can be accessed on reading log (link above)
* Main takeaway from this paper:
> Branch prediction can be improved by replacing the predictor’s pattern history table (PHT) with a set of perceptrons (simple neural networks), which can learn as the program runs. Rather than try to improve the predictor by targeting aliasing, the perceptron approach lowers misprediction rates by increasing the accuracy of the prediction algorithm itself.
 * Can go back and look over some extra terms just for fun later

**Friday, Oct 12 (1 hr)**

* Had first meeting with Professor Sherwood and Deeksha at ArchLab today, asked a bunch of questions about the paper we read, the work we'd be doing, and the tools we'll be using. Here is a brief summary of the notes I took during the meeting (there was a lot going on, so these might not be phrased very precisely, but they should be a useful overview):
   * We'll be involved some kind of chip design (programming it, not hardware) that can run neural networks and maximize efficiency by minimizing energy use (we'll develop our own specific research question though)
   * Main focus: Interaction between energy and machine, how adjusting hyperparameters of neural network can change performance
   * We'll be using PyRTL, so learn how to use it. This means doing the whole bash Ubuntu setup so I can work in the terminal pretty much normally on Windows 10.
   
* Terms to look up:
  * ASIC
  * TensorFlow
  * PyTorch
---
---
# Week 2 (10/4- 10/10)
**Goals:**
- [x] Read Research Paper Prep by Friday (take notes)
- [x] Read assigned research paper (Dynamic Branch Prediction...)
- [x] Schedule meetings with Professor Sherwood, research group, and Mai
- [x] Edit group page- add meeting times and link to log
- [x] Make digital reading log

   - [x] Clean and format reading log

**Side Stuff**
- [ ] Start learning Python
- [ ] Do some reading on the hardware stuff mentioned in papers

**Tuesday, Oct 9 (1.5 hrs)**

* Finished reading "Dynamic Branch Prediction...." first pass, finished notes on results and main body of the paper. Overall, as I understand it, the paper talks about applying a simple neural network, the perceptron, to reduce the percentage of mispredictions of branches, which slow down performance. 
* ["Perceptron" notes](https://docs.google.com/document/d/1w88Sf5nNbMch-iYMC7N0dAdIlIEadZRttGgzu_mLrHc/edit#bookmark=id.4tfbk42ka5g)
* ["Pythonic Approach" notes](https://docs.google.com/document/d/1w88Sf5nNbMch-iYMC7N0dAdIlIEadZRttGgzu_mLrHc/edit#bookmark=id.sm3jmho09qam)
**Monday, Oct 8 (2 hrs)**

* Met up with Maggie and started doing a second pass on "A Pythonic Approach...", since we had both read it over the weekend. Discussed main ideas and looked up terms we didn't understand. Shortly after, we learned that we weren't actually supposed to read that paper, and we were assigned a new one. It was still an interesting introduction to the PyRTL tool, and it gave me an idea of some of the hardware stuff going on that I have to learn about.

* Started reading the actual assigned reading, "Dynamic Branch Prediction with Perceptrons"- went over abstract, intro, and experiments and results sections. I have set up a digital reading log on a separate Google Docs, linked [here] (https://docs.google.com/document/d/1w88Sf5nNbMch-iYMC7N0dAdIlIEadZRttGgzu_mLrHc/edit?usp=sharing). It's still pretty messy, so I'm planning to clean that up and add in my notes on "A Pythonic Approach" because why not. 

**Sunday, Oct 7 (2.5 hrs)**

* Started reading assigned Computer Architecture paper- "A Pythonic Approach for Rapid Hardware Prototyping and Instrumentation" and answering questions about each part. So far, paper seems to be about the implementation of the PyRTL language as an easier way to do hardware design without requiring a deep knowledge of hardware, or the steep learning curve of traditional hardware design languages (HDLs). Will be continued tomorrow, with more notes.

**Saturday, Oct 6 (2.5 hrs)**

-To Do-

  * Get started on reading the assigned research paper before the end of the weekend

-Things I Did-

* Read "How to Read a Research Paper" and took notes. Takeaways: Marking up papers is very helpful, keep notes as you answer the 5 key questions
  * I'm thinking as we read different papers, we could keep a list of the summaries of each along with their citations so we have them ready to go when we want to use them
* Met up with team for dinner- We've officially scheduled our team meetings for Wednesdays 1:00pm-3:00pm at the library, though we can meet for more time later on. Location is fairly flexible as well. Also contacted Mai and Professor Sherwood to set up weekly meetings. If all goes as planned, our meetings with Mai will be on Wednesday right before class (3:00-3:30pm), and we'll meet Sherwood and/or his grad students sometime on Friday. His lab hours are also on Friday, but I'm not sure when, so we'll have to fit the meeting in around the labs and our classes. 
* Updated the group page with group meeting times and link to my research log. In hindsight, I probably should've just added links to everyone's logs at once, but I didn't know I would have to wait for the merge request to be accepted oops.
---
---
# Week 1 (9/27-10/3)
**Goals:**

- [x] Set up research log

- [x] Attend research group meeting on Friday when scheduled, record attendance

- [x] Record ERSP preliminary thoughts

- [x] Reflect on research logs

- [x] Get started on reading research paper for Team Sherwood for next Wednesday

**Wednesday, Oct 3 (2 hrs)**

-To Do-
  * Schedule a weekly time for group to meet
  
 -Important Things I Did Today-
 
 * Built the noicest spaghetti house this side of Pastatown. It didn't actually stand (at all), but it was a real team effort. It might have been better if we had actually slowed down and planned it out first, so note taken there, but overall it was really exciting to get to know my teammates through this challenge. I think we did a good job of listening to everyone's ideas, picking the one we thought sounded best, and making sure everyone was involved in the construction. Next time, though, we definitely need a better foundation. Those pasta legs were not doing it.
 
 * Read the rest of the intro to ERSP letters from past students- learned importance of time management and working with team members and grad students
 
 * Formed a group chat with teammates Maggie, Dylan, and Dawit. We're working on finding a good time to meet every week (probably Fridays), though of course this may shift around as the project progresses. Dylan created an online calendar we all put our free times into, so that's a good weekly reference.
 [When to Meet? Calendar](https://www.when2meet.com/?7163758-lOIkc)
 

**Tuesday, Oct 1 (1.5 hrs)**

* Set up log
* ERSP initial thoughts
* Log reflection

*ERSP Initial Thoughts*

1. This is what I'm most excited about in ERSP:

I'm most excited to learn the process of research and work in a team to develop a project. I'm looking forward to learning a new subject and, at the same time, apply it to a specific problem. While I'm relatively new to computer science in general, I feel like I've spent a long time on foundational material without actually applying it in a meaningful setting. It's exciting to think about how I'll be collecting real information and learning how to write an academic engineering paper. I think, with ERSP, I'll feel more experienced and confident about my ability to contribute to the tech world.

2. This is what I'm most nervous about in ERSP:

I don't have any experience in computer architecture, and I'm worried I'll have to struggle to keep up when I'm introduced to these new topics. Research in general seems like an intimidating and cryptic process, so while I'm excited to have the opportunity to unravel it, I'm also concerned that I won't work well enough to support the rest of my team members.

*Log Reflection*

1. How did the logs differ in style? What advantages are there in one style over another?

In terms of layout, Miranda Parker's log was definitely a lot more advanced (expecially the organized intro with important links at the top), while Adrian Mendoza's was more basic and linear. They both had a similar format, however, starting each week with a list of goals and discussing what they accomplished every day. Jessica's writing was very casual, while Adrian's was more concise. I think Jessica's style has some advantages in that she presents a very clear summary of the project up front with some of the material that was important to the project,and her writing style was very readable. Adrian's is less 'flashy' but, since it was done as part of an ERSP project as well, seems more familiar. It's also interesting to see his progress in research and the ERSP program itself over the course of his log.

2. How  were the logs useful to the researcher and those working with the researcher?

The logs both kept very careful track of what the researcher did that day, where they found the data, and who they worked with that day. Obviously, these records are useful to the researcher because they have an opportunity to review their work and determine if they're making adequate progress. (The "Goals" section each week seems especially important for keeping them on track.) Those who are working with the researcher can read these logs to get a good idea of what problems they've been targeting and what skills they have. The logs could also be useful for catching up on each others' work if the group doesn't meet for a day or two.

3. Did the students keeping the logs seem to meet their goals? Did they get better at meeting their goals over time?

I definitely feel like keeping the logs helped the students stay focused: listing their goals each week and tracking their own progress everyday seemed to help them determine whether they were doing enough to stay on top of their projects. From the two example logs I read, it seemed like the students pretty much always accomplished what they set out to do each week, but they started setting more ambitious goals for themselves over time.

4. Anything else to say about logs?

I really liked reading Miranda Parker's log- it's full of so much cool information but it's also really personable and casual. It seemed like she enjoyed her work. Adrian's was very helpful, since it gave me an idea of the scale and workload of the ERSP projects.


##Important Links:
[*My Reading Log*](https://docs.google.com/document/d/1w88Sf5nNbMch-iYMC7N0dAdIlIEadZRttGgzu_mLrHc/edit)

# Week 3 (10/11-10/17)
**Goals:**
- [x] First meeting with Professor Sherwood this Friday- ask about lab hours
- [ ] Set up Ubuntu on windows to use terminal for Python
- [ ] At least download PyRTL tool, hopefully start working with it
- [ ] Make another pass over prev paper ("Dynamic Branch Prediction...")
- [ ] Read new paper when Deeksha/ Professor Sherwood sends it to us

**Sunday, Oct 14 (2hrs)**
* Read second pass on "Dynamic Branch Prediction....", filled out questions for part 2 of reading assignment. Can be accessed on reading log (link above)
* Main takeaway from this paper:
  Branch prediction can be improved by replacing the predictor’s pattern history table (PHT) with a set of perceptrons (simple neural networks), which can learn as the program runs. Rather than try to improve the predictor by targeting aliasing, the perceptron approach lowers misprediction rates by increasing the accuracy of the prediction algorithm itself.
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
- [ ] Read assigned research paper (Dynamic Branch Prediction...)
- [x] Schedule meetings with Professor Sherwood, research group, and Mai
- [x] Edit group page- add meeting times and link to log
- [x] Make digital reading log

   - [ ] Clean and format reading log

**Side Stuff**
- [ ] Start learning Python
- [ ] Do some reading on the hardware stuff mentioned in papers

**Tuesday, Oct 9 (1.5 hrs)**

* Finished reading "Dynamic Branch Prediction...." first pass, finished notes on results and main body of the paper. Overall, as I understand it, the paper talks about applying a simple neural network, the perceptron, to reduce the percentage of mispredictions of branches, which slow down performance. 

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


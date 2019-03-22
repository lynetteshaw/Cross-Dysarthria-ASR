# Model Proposal for Acoustic Analysis for Huntington Disease Classification

Matthew Perez

* Course ID: CMPLXSYS 530,
* Course Title: Computer Modeling of Complex Systems
* Term: Winter, 2019



&nbsp; 

### Goal 
*****
 
My goal is to study whether specific audio cues and acoustic patterns can be extracted and used to characterize Huntington Disease (HD). I plan to use audio recordings of participants reading the grandfather passage and will investigate acoustic cues which capture muscaltory failure which is a common symptom of HD. Lastly, I will see how effective these features are at characterizing disease through classification.

&nbsp;  
### Justification
****
ABM is perfect for this type of research as it helps explore the effectiveness of specific features captured. Building models such as Random Forest or Neural Networks exhibit the agent-based quality for ABM because each classifier can be thought of a specific agent in a system. The goal of these multi-agent systems is to provide class posteriors and hopefully correctly classify the class of the resulting test data using input features. I plan to have these agents "interact" together through ensembling, where each agent casts a vote on the specific class output and the majority vote is used as the final class output of the multiagent system. The advantage of this apporach is that more of the search space can be explored (through different parameters, initializations, etc.) through having multiple agents.

*LS COMMENTS: Very interesting direction for this project! If you haven't already looked into "Learning Classifier Systems", the literature in that area might be of interest to you (though you by no means need to try a develop a full system for this course, unless you want to...ensemble methods will be close enough for our purposes). Interesting side note if you don't already know, John Holland who first laid these out in 1980 was a founder of Complex Systems here at Michigan :)*

&nbsp; 
### Main Micro-level Processes and Macro-level Dynamics of Interest
****

I'm really interested in how these features change with respect to disease severity (hopefully they become more discriminative with disease progression). Also, I'm interested in how these features change over time. Using time series analysis can I detect changes such as fatigue within the features extracted over the short passage reading.

&nbsp; 


## Model Outline
****
&nbsp; 
### 1) Environment
No environment. Maybe in my setup the environment is extracting features?
Currently the features which I am investigating are called shapelets, which are pitch segments patterns that ideally cluster based on class (still researching this at the moment). My feature if this works will be the distance of a given test shapelet to the closest set of neighbors (k-NN) in the corresponding shapelet space. The downside: The time complexity for generating these shapelets is massive!

*LS COMMENTS: Input data and data used to validate predictions is sometimes construed as being the environmental, albeit in a very abstract manner. Interesting hypothesis!*

```python

def main():
    prosody_path = sys.argv[1]
    word_ali_file = sys.argv[2]
    features_file = sys.argv[3]

    pitch_path = os.path.join(prosody_path,'pitch.txt')
    energy_path = os.path.join(prosody_path,'energy.txt')

    #Raw pitch and energy
    raw_pitch, raw_energy = read_pitch_energy(pitch_path, energy_path)

    # Tunable Parameters = shapelet length, threshold, shapelet redundancy value 
    # sweep shapelet length
    for shapelet_len in [100, 10]:
        pitch_shapelet = {}
        energy_shapelet = {}
        for spkr in raw_pitch:
            # Compute shapelets from raw frames (i.e. no segmented info like where phone/word is)
            pitch_shapelet[spkr] = compute_shapelet_frame(raw_pitch[spkr], shapelet_len)
            # energy_shapelet[spkr] = compute_shapelet_frame(raw_energy[spkr], shapelet_len)


        all_spkrs = late_balanced.keys()
        shapelets, IG, shapelet_class = cluster_shapelets_pool(pitch_shapelet, all_spkrs)


```

&nbsp; 

### 2) Agents
Number of agents: 10
Number of features: unknown
List of features: distance-to-nearest-shapelet, neighboring-shapelets-class
agent methods: init, fit, predict

Considerations for agents: Train these agents using a balanced dataset. Small amounts of data is available for this research, investiagte methods such as K-Fold-Cross-Validation in order to cycle test/train data and maximize data available.

*LS COMMENTS: good approach.*

```python
#Extract Features
#Read features into dataframe

# Classify using randomForest
clf = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=2)
clf.fit(X_tr, Y_tr)
predictions = clf.predict(X_te)
predict_probs = clf.predict_proba(X_te)

acc = accuracy_score(Y_te, predictions)
uar = recall_score(Y_te, predictions, average='macro')
```

&nbsp; 

### 3) Action and Interaction 
Agents will run indepedently (i.e. be trained and predict independently) and will interact solely at the end when tehir class predictions are shared. The mode (majority) prediction will be used as the final class output.

_What does an agent, cell, etc. do on a given turn? Provide a step-by-step description of what happens on a given turn for each part of your model_

Step 1: a given agent will be trained using the training data given.
Step 2: After training, will be performed by predicitng class output of held-out validation data (originally from the training set)
Step 3: New parameters will be initialized and steps 1-2 will be repeated
Step 4: Best parameters will be used for the final test set
Step 5: Predict class output for a given test set
Step 6: repeat steps 1-5 for N agents
step 7: Repeat steps 1-6 for M simulations

&nbsp; 
*LS COMMENTS: Solid setup. Will steps 1-2 be repeated multiple time to see if you get some convergence in which parameters to use? Also, see prior comment about learning classifier systems for potential inspiration on ways of having agents/classifiers interact with one another or selection of.*

### 4) Model Parameters and Initialization

Refer to "Action and Interaction" above.
Model parameters include things such as learning rate and optimizer for neural network classifiers and max tree depth for random forest classifiers.

&nbsp; 

*LS COMMENTS: Time allowing, you might also consider varying the number of individual classifiers you train as well.*

### 5) Assessment and Outcome Measures

Since I am measuring multi-class classification, I will use unwieghted average recall (UAR) as my metric for evaluation. Additionally I will also be measuring pearson's correlation in order to see the impact of individual features and if the correlations of specific features are reasonable (i.e. negative correlation between speech rate and disease progression).

&nbsp; 

### 6) Parameter Sweep


_What parameters are you most interested in sweeping through? What value ranges do you expect to look at for your analysis?_

I will be sweeping through the the tunable parameters for optimizing my classifiers. However, I am most interested in sweeping through multiple different features and perhaps using feature selection in order to reduce overfitting and counteract the curse of dimensionality (overcomplciating my system). I expect specific features to perform better than others (i.e. expect speech rate extracted from transcripts to be more effective than acoustic features extracted acoustically since there is a lot more noise in acoustics. However, there is less assumptions and requirements (i.e. transcriptions) for acoustic-based features and hopefully they can provide a detailed account of specific patterns happening in speech.

*LS COMMENTS: Solid setup! Look forward to seeing where this project goes. 19/20*


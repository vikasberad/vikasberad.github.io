---
permalink: /
title: "Scalable and Accurate Deep Learning for Electronic Health Records"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Predictive modeling using Electronic Health Records (EHR) data holds immense potential for advancing personalized medicine and enhancing healthcare outcomes. However, traditional approaches demand intensive data extraction and curation, often discarding valuable information. Rajkomar et al.'s research presents a scalable and accurate deep learning framework that directly leverages raw EHR data for clinical predictions. This blog delves into their findings, methods, and the implications for healthcare.

The Challenges of Traditional EHR Predictive Models
======

Traditional models require the manual extraction of structured predictor variables.

A significant amount of clinical data, particularly from unstructured sources like clinical notes, remains unused.

The labor-intensive process limits scalability and model accuracy.

Rajkomar et al. addressed these challenges by adopting an entirely novel approach that bypasses the need for extensive data preprocessing.

Leveraging FHIR Data Representation
======
The study employed the Fast Healthcare Interoperability Resources (FHIR) format to represent patient records. This approach organizes healthcare data in chronological order without the need for site-specific harmonization.

Key Features of FHIR Data Representation:

Temporal Ordering: Ensures that events in a patient's timeline are accurately recorded.

Comprehensive Inclusion: Incorporates structured data, free-text notes, and other clinical observations.

Standardized Format: Makes it easier for models to learn from diverse data sources.

Deep Learning Model Architectures
------
The authors explored three neural network models to process EHR data:

Recurrent Neural Networks (RNNs) with LSTM: Handle temporal dependencies in the data.

Time-Aware Attention Neural Networks (TANNs): Focus on relevant time-sensitive information.

Boosted Time-Based Decision Stumps: Capture binary decision rules for data partitioning.

Ensemble Approach

To optimize accuracy, the final predictive model combined the outputs from all three neural network architectures.

Create content & metadata
------
For site content, there is one markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For example, each talk is a markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory).

**Markdown generator**

The repository includes [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual markdown files that will be properly formatted for the Academic Pages template. The sample CSVs in that directory are the ones I used to create my own personal website at stuartgeiger.com. My usual workflow is that I keep a spreadsheet of my publications and talks, then run the code in these notebooks to generate the markdown files, then commit and push them to the GitHub repository.

How to edit your site's GitHub repository
------
Many people use a git client to create files on their local computer and then push them to GitHub's servers. If you are not familiar with git, you can directly edit these configuration and markdown files directly in the github.com interface. Navigate to a file (like [this one](https://github.com/academicpages/academicpages.github.io/blob/master/_talks/2012-03-01-talk-1.md) and click the pencil icon in the top right of the content preview (to the right of the "Raw | Blame | History" buttons). You can delete a file by clicking the trashcan icon to the right of the pencil icon. You can also create new files or upload files by navigating to a directory and clicking the "Create new file" or "Upload files" buttons. 

Example: editing a markdown file for a talk
![Editing a markdown file for a talk](/images/editing-talk.png)

For more info
------
More info about configuring Academic Pages can be found in [the guide](https://academicpages.github.io/markdown/), the [growing wiki](https://github.com/academicpages/academicpages.github.io/wiki), and you can always [ask a question on GitHub](https://github.com/academicpages/academicpages.github.io/discussions). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.

# MSA_ML_Tutorial2024
The materials for the MSA Tutorial on ML for Microscopy

Machine Learning for Electron Microscopy: from Data Analysis to Active Experiments
Sergei V. Kalinin, UT Knoxville and PNNL
Kevin M. Roccapriore, AtomQ and Oak Ridge National Laboratory

8.30 – 9.00 	Lecture 1: Machine learning in full electron microscopy workflow (Kalinin)

9.00 – 10.00 	Lecture 2: ML for spectroscopy and imaging data (Kalinin)

10.00 – 10.30 	Coffee break

10.30 – 12.00 	Lecture 3: Variational Autoencoders for imaging, spectra, and structure-property 			relationships (Kalinin)

12.00 – 1.00 	Lunch break

1.00 – 2.00 	Lecture 4: From post-acquisition to real-time analysis: ensemble networks and 			automated experiment (Roccapriore)

2.00 – 2.30 	Coffee break

2.30 – 3.30 	Lecture 5: Decision making algorithms and reward-based workflows (Kalinin)

3.30 – 4.30 	Lecture 6: Automated EELS and 4D STEM (Roccapriore)

4.30 – 5.00 	Lecture 7: Human in the loop automated experiment and LLM-based co-scientists 			(Kalinin)


Overview:
Machine Learning (ML) and Artificial Intelligence (AI) are transforming various fields, including medicine, autonomous driving, and now, electron microscopy. Over the past two years, the barriers to adopting and implementing ML methods in microscopy have significantly lowered, thanks to advancements like code assistants and Python APIs provided by many manufacturers. Despite this, ML is often perceived as a collection of disparate tools. This course aims to integrate these tools and demonstrate how ML can be seamlessly used across all aspects of a microscopist's work, from sample selection and instrument optimization to data analysis.

Course Objectives:
•	Introduction to ML Pipelines: Learn the basics of ML pipelines for image and spectrum analysis, covering classical ML tools, deep networks, and variational autoencoders.
•	Transition to Real-Time Analysis: Understand the shift from post-acquisition to real-time analysis, allowing human operators to make decisions in high-sampling or high-dimensional streaming data scenarios.
•	Real-Time Experimentation: Explore the incorporation of ML methods in real-time decision-making, discussing requirements such as knowledge loops (KL), human-in-the-loop experimentation, and the role of large language models (LLM) based co-scientists.

The tentative course outline is:
1.	Machine learning methods along the microscopy workflow: 
•	Classical workflows: sample selection, microscope optimization, post-acquisition analysis
•	ML for real time analysis
•	ML for Automated experiment
2.	Building the image and spectrum analysis pipelines:
•	Learning from image and spectral data: linear methods
•	Reward-based workflow design
•	Deep learning for image segmentations
3.	Variational Autoencoders and encoders-decoders
•	Principle of VAEs
•	VAEs denoising and superresolution: opportunities and pitfalls
•	VAE for physics discovery and library construction
•	Building structure-property relationships
4.	Machine learning for real-time data analysis
•	DCNNs for microscopy
•	Ensemble networks
•	Reward driven workflows
5.	Active learning and optimization
•	Principles of decision making
•	Gaussian Processes and Bayesian Optimization
•	Instrument optimization
•	Learning structure-property relationships in automated fashion
6.	Automated experiment in STEM-EELS and 4D STEM
•	Deep kernel learning (DKL)
•	Hyperspectral predictions from sparse sampling
•	Scalarizers that guide autonomous experiments
•	From single loop to mixture of experts and gated networks
7.	Human in the loop automated experiment and LLM co-scientists
•	Workflows with shifting rewards.
•	Exploring beyond-human workflow design


Why This Course?
The course explores how ML can be used for all tasks in a microscopist's workflow. With the recent advancements in code assistants and accessible Python APIs, the timing is perfect to dive into ML for electron microscopy. This course is designed to provide a comprehensive understanding of ML principles and methods while focusing on practical applications in the lab.

Key Takeaways:
•	Master ML Integration: Learn to use ML for tasks ranging from sample selection to instrument optimization and data analysis.
•	Deep Dive into ML Tools: Learn about classical ML tools, deep networks, and variational autoencoders.
•	Real-Time Analysis Skills: Develop skills to transition from post-acquisition to real-time analysis, crucial for high-dimensional data.
•	Incorporate ML in Experiments: Understand the integration of ML in real-time decision-making and experimentation.

Embark on an exciting journey to explore ML's potential in electron microscopy. Discover how ML can revolutionize your work, unraveling complex problems and uncovering innovative solutions. By enrolling, you will:
•	Master the synergy of domain-specific knowledge, ML understanding
•	Learn the pathway for effective coding mastery.
•	Transform real-world problems into ML solutions.
•	Explore groundbreaking advancements in electron microscopy.
•	Shape the Future with ML!

Instructor Experience:
Sergei V. Kalinin is a Weston Fulton chair professor at the University of Tennessee, Knoxville and Chief Scientist for AI/ML in Physical Sciences, PNNL. In 2022 – 2023, he has been a principal scientist at Amazon (special projects). Before then, he had spent 20 years at Oak Ridge National Laboratory where he was corporate fellow and group leader at the Center for Nanophase Materials Sciences. He received his MS degree from Moscow State University in 1998 and Ph.D. from the University of Pennsylvania (with Dawn Bonnell) in 2002. He has >15 years of experience applying AI in experimental physical sciences from data analytics, theory-experiment matching, and automated and autonomous microscopy and materials synthesis. His research focuses on the applications of machine learning and artificial intelligence methods in materials synthesis, discovery, and optimization, automated experiment and autonomous imaging and characterization workflows in scanning transmission electron microscopy and scanning probes for applications including physics discovery, atomic fabrication, as well as mesoscopic studies of electrochemical, ferroelectric, and transport phenomena via scanning probe microscopy. Sergei has co-authored >650 publications, with a total citation of ~50,000 and an h-index of >115. He is a fellow of AAIA, MRS, APS, IoP, IEEE, Foresight Institute, and AVS; a recipient of the Feynmann Prize of Foresight Institute (2022), Blavatnik Award for Physical Sciences (2018), RMS medal for Scanning Probe Microscopy (2015), Presidential Early Career Award for Scientists and Engineers (PECASE) (2009); Burton medal of Microscopy Society of America (2010); 5 R&D100 Awards (2008, 2010, 2016, 2018, and 2023); and a number of other distinctions.
Kevin Roccapriore is a staff scientist at Oak Ridge National Laboratory (ORNL) within the Center for Nanophase Materials Sciences (CNMS). Kevin will be pioneering a solid-state quantum computing-based start-up called AtomQ, which is based on atomic manipulation of three-dimensional materials. Starting August 2024, he will be a fellow in Innovation Crossroads, which is a Department of Energy (DOE) Lab Embedded Entrepreneurship Program (LEEP) designed to jump-start and accelerate energy-based startups. He received his MS and PhD in Physics from the University of North Texas in 2018, followed by his post-doctoral appointment at ORNL and then staff position starting in 2021. During his time at ORNL, Kevin developed several AI-based workflows integrated directly on electron microscopes for scientific discovery and exploration. He is a PI of the INTERSECT Initiative at ORNL, which bridges electron microscopy instrumentation to high performance computing. His interests are in light-matter interaction and optical properties of materials, how atomic and nano-scale structures play a role in these properties, and finally, how they can be manipulated at the smallest of scales. His research focus therefore has been in studying low dimensional materials using various modalities of the scanning transmission electron microscope (monochromated EELS, 4D-STEM) with machine learning and artificial intelligence – first in post-processing analysis routines, then  during microscope operation via automated and autonomous experiments. Kevin has co-authored >30 publications and is a recipient of the 2023 R&D100 Award and 2024 Microscopy Today Innovation Award.

---
layout: post
title:  "Use cases of differential privacy"
author: ria
categories: [ differential-privacy, application ]
image: assets/images/13.jpg
---

In this blog post, we will cover use cases of differential privacy ranging from healthcare to geolocation. For the slide deck associated with this post, please see [Use cases of Differential Privacy and Federated Learning by @Ria](https://docs.google.com/presentation/d/15Mzb0mGKrBSDULTuha-TXHp-rdHppLi8MQGTuiwfKlU/edit?usp=sharing)

#### Genomics

Biggest risk: linkage attacks using auxiliary information

> *“It has been demonstrated that even coarse-level information such as minor allele frequencies (MAF) can reveal whether a given individual is part of the study cohort, potentially disclosing sensitive clinical phenotypes of the individual.”* [^fn1]



Prior solutions[^fn2]:

- De-identification: lose meaningful info
- K-anonymization: offers no formal privacy guarantees

The benefits associated with Differential Privacy [^fn2]:

- Protects against linkage attacks
- Interactive setting: Query non-public database - answers are injected with noise or only summary statistics are released
- Non-interactive setting: Public data injected with noise
- Disadvantage: Privacy vs. Utility
- Only preset queries allowed: ‘return p-value’, ‘return location of top K SNPs’

#### Uber

##### Definitions of Sensitivity

*Sensitivity of a query:* Amount query’s results change when database changes. 
*Global sensitivity:* Maximum difference in the query’s result on any two neighboring databases.
*Local sensitivity:* Maximum difference between the query’s results on the true database and any neighbor of it.Local sensitivity is often much lower than global sensitivity since it is a property of the single true database rather than the set of all possible databases.Smoothing functions are important.
Many differential privacy mechanisms are based on global sensitivity, and do not generalize to joins (since they can multiply input records).
Techniques using local sensitivity often provide greater utility, but are computationally infeasible.

##### Use case

Example: Determine average trip distance[^fn9]

The authors propose Elastic Sensitivity as a method to leverage local sensitivity.

smaller cities might have fewer trips so an individual trip is likely to influence the analysis more

The purpose is to “model the impact of each join in the query using precomputed metrics about the frequency of join keys in the true database”.

The authors demonstrate FLEX, a system that utilizes elastic sensitivity; benefits described in the paper:

- Provides (ε, δ)-differential privacy and does not need to interact with the database.
- Only requires static analysis of the query and post-processing of the query results.
- Scales to big data while incurring minimal performance overhead.

![img](https://lh6.googleusercontent.com/DpeS5uq9fjKTlT9lG5Ke4hFnF-MxzS5iiG4ospYsCwrrDpU_jF4EktuYVlEEPRCbL_VxTIaMuYTzTAsMXpFCW8VrT54q8W5RuOJoJa0sZWXqavXPPhg5P3Rk1m4I2JXUWWH_)

![img](https://lh5.googleusercontent.com/-UMB6w6XmQNrGoXobcn4Mo1mzDFD27ymYVnuWwDKCBQMTYfXoyTuGFiioNHtKOhXIPtcsVxad9tT1vAycO5ULQoG34SloBxVuYZh5H3pbVUgbmIN3mebudaS-6BYiFjR2heT)

*Definition: “equijoins are joins that are conditioned on value equality of one column from both relations.”*

![img](https://lh4.googleusercontent.com/RPzHz--3UOg57AP8ucmvBvTsBEsuMGsU7bY8e4CyADltqN1d0BTXaVyFNwoQd77DGnkmszTrQib1Mr-Zr6OzcQwcO2_8mbF4XcaHqKOz8NKWDi2nsdHpTBfDTulzmGrHoJIB)

#### Healthcare + Internet of Things

Collect health data streams measured at fixed intervals(e.g. collecting heart rates measured every minute during business hours)3

Perturb data using Local Differential Privacy, where data contributor adds noise

![img](https://lh6.googleusercontent.com/X93uPa9za6kNKEPjejKsQHWMLX7w96gW1yLEj_xERkMiEDrD147G6Fk2buFBtEu2xhMaHahm-5FV8zDwp1RJFaYAywhNlLOBDMXYQzYbdYuSvTWYx8x0XECi7k7WHHMAXprw)



#### Healthcare

Visualizing or handling large data for biomedical applications with differential privacy guarantees

...



**Differential Privacy References**

[^fn1]: [[1\] Machine learning and genomics: precision medicine versus patient privacy](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2017.0350?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed&)
[^fn2]: [[2\] Emerging technologies towards enhancing privacy in genomic data sharing](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1741-0)
[^fn3]: [[3\] Privacy-preserving aggregation of personal health data streams](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207639)
[^fn4]: [[4\] Demonstration of Damson: Differential Privacy for Analysis of Large Data ](http://differentialprivacy.weebly.com/uploads/9/8/6/2/9862052/pid2574139.pdf)
[^fn5]: [[5\] Compressive Mechanism](https://differentialprivacy.weebly.com/compressive-mechanism.html) 
[^fn6]: [[6\] Project PrivTree: Blurring your “where” for location privacy](https://www.microsoft.com/en-us/research/blog/project-privtree-blurring-location-privacy/)
[^fn7]: [[7\] A History of Census Privacy Protections](https://www.census.gov/library/visualizations/2019/comm/history-privacy-protection.html)
[^fn8]: [[8\] Protecting the Confidentiality of America’s Statistics: Adopting Modern Disclosure Avoidance Methods at the Census Bureau ](https://www.census.gov/newsroom/blogs/research-matters/2018/08/protecting_the_confi.html)
[^fn9]: [[9\] Towards Practical Differential Privacy for SQL Queries](https://arxiv.org/pdf/1706.09479.pdf)
[^fn10]: [[10\] Privacy-preserving biomedical data dissemination via a hybrid approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371369/pdf/2977168.pdf)

Additional resource: [Differential privacy: its technological prescriptive using big data](https://link.springer.com/content/pdf/10.1186/s40537-018-0124-9.pdf)

**Differential Privacy Code Repositories**

- [Uber SQL Differential Privacy](https://github.com/uber-archive/sql-differential-privacy)
- [TensorFlow - Differential Privacy](https://blog.tensorflow.org/2019/03/introducing-tensorflow-privacy-learning.html?m=1)
- [Google’s C++ Differential Privacy library](https://github.com/google/differential-privacy)
- [OpenMined Differential Privacy](https://blog.openmined.org/making-algorithms-private/)

Stay posted for a future blog post on use cases for federated learning!
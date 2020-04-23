---
layout: post
title:  "Use cases of differential privacy"
author: ria
categories: [ differential-privacy, application ]
image: assets/images/13.jpg
---

In this blog post, we will cover use cases of differential privacy (DP) ranging from healthcare to geolocation. For the slide deck associated with this post, please see [Use cases of Differential Privacy and Federated Learning by @Ria](https://docs.google.com/presentation/d/15Mzb0mGKrBSDULTuha-TXHp-rdHppLi8MQGTuiwfKlU/edit?usp=sharing)

#### Genomics

Machine learning has important implications for genomics applications, such as for precision medicine (i.e. treatment  tailored to a patient's clinical/genetic features) [^fn1] and detecting fine-grained insights in data collected from a diverse population [^fn2].

Given the rapid creation of numerous genomics datasets to fuel statistical analyses and machine learning research for these applications, one of the primary privacy risks for such an application are linkage attacks using auxiliary information. Linkage attacks involve exploiting the scenario where information in a public database overlaps with a sensitive dataset (which is usually anonymized/de-identified to censor the dataset). We'll cover de-identification and k-anonymization in a moment. 

There are many illustrated examples of linkage attacks, such as a linkage attack being deployed on de-identified hospital records and a voter registration database to find fhe Governor of Massachusetts's patient profile [^fn2].

Furthermore, consider the following quote: 

> *“It has been demonstrated that even coarse-level information such as minor allele frequencies (MAF) can reveal whether a given individual is part of the study cohort, potentially disclosing sensitive clinical phenotypes of the individual.”* [^fn2]

This is concerning in light of genetic discrimination, where individuals can be treated differently because they might have a genetic mutation [^fn1].

Prior solutions to this issue include [^fn1]:

- De-identification, which involes removing unique identifiers from the data such as names, phone numbers, and even vehicle identifiers. The disadvantage of this approach is that you could lose meaningful information.
- K-anonymization, which involves removing information from the released data until a data record belong in the same equicalenc class as at least (k − 1) other records. The disadvantage of this approach is that it offers no formal privacy guarantees and is vulnerable to linkage attacks, among other attacks. 

The benefits associated with Differential Privacy [^fn1]:

* Protects against linkage attacks

* Enables two types of settings:
  * Interactive setting: Query non-public database - answers are injected with noise or only summary statistics are released
  * Non-interactive setting: Public data injected with noise

* Disadvantages:
  * Balancing Privacy vs. Utility (i.e. considering the accuracy of the results).
  * Only preset queries are allowed with DP approaches such as: ‘return p-value’, ‘return location of top K SNPs’

#### Uber User Data

Before discussing the use case, let's quickly define the different types of sensitivity for a query.

##### Definitions of Sensitivity [^fn9]: 

* *Sensitivity of a query:* Amount query’s results change when database changes. 

* *Global sensitivity:* Maximum difference in the query’s result on any two neighboring databases.

* *Local sensitivity:* Maximum difference between the query’s results on the true database and any neighbor of it. Local sensitivity is often much lower than global sensitivity since it is a property of the single true database rather than the set of all possible databases. Smoothing functions are important to consider.

Many differential privacy mechanisms are based on global sensitivity, and do not generalize to joins (since they can multiply input records).

Techniques using local sensitivity often provide greater utility, but are computationally infeasible.

##### Use case

For this use case, a sample application by Uber is to determine average trip distance for users [^fn9]. Smaller cities might have fewer trips so an individual trip is likely to influence the analysis, which differential privacy can help address.

Per the notes frmo the previous section, it is valuable to consider local sensitivity given global sensitivity-based DP mechanisms do not geeneralize to joins. The below image from the paper "Towards Practical Differential Privacy for SQL Queries" [^fn9] shows a large number of queries utilize joins, which motivates the need for a method that takes advantage of local sensitivity. Side note: I highly recommend reading the paper "Towards Practical Differential Privacy for SQL Queries" for similar analyses of queries, and a detailed definition of Elastic Sensitivity.

![img](https://lh6.googleusercontent.com/DpeS5uq9fjKTlT9lG5Ke4hFnF-MxzS5iiG4ospYsCwrrDpU_jF4EktuYVlEEPRCbL_VxTIaMuYTzTAsMXpFCW8VrT54q8W5RuOJoJa0sZWXqavXPPhg5P3Rk1m4I2JXUWWH_)

The authors propose Elastic Sensitivity as a method to leverage local sensitivity. The purpose of the approach is to “model the impact of each join in the query using precomputed metrics about the frequency of join keys in the true database”. Please see the below table for a comparison between Elastic Sensitivity with other DP mechanisms - we see Elastic Sensitivity supports different types of equijoins, which "are joins that are conditioned on value equality of one column from both relations."


![img](https://lh5.googleusercontent.com/-UMB6w6XmQNrGoXobcn4Mo1mzDFD27ymYVnuWwDKCBQMTYfXoyTuGFiioNHtKOhXIPtcsVxad9tT1vAycO5ULQoG34SloBxVuYZh5H3pbVUgbmIN3mebudaS-6BYiFjR2heT)


The authors demonstrate FLEX, a system that utilizes elastic sensitivity. Here are the benefits described in the paper:

- Provides (ε, δ)-differential privacy and does not need to interact with the database.
- Only requires static analysis of the query and post-processing of the query results.
- Scales to big data while incurring minimal performance overhead.

![img](https://lh4.googleusercontent.com/RPzHz--3UOg57AP8ucmvBvTsBEsuMGsU7bY8e4CyADltqN1d0BTXaVyFNwoQd77DGnkmszTrQib1Mr-Zr6OzcQwcO2_8mbF4XcaHqKOz8NKWDi2nsdHpTBfDTulzmGrHoJIB)

#### Healthcare + Internet of Things: Heartrate monitoring

Let's now turn to a healthcare application involving wearable technology and the Internet of Things. The use case here is to collect health data streams measured at fixed intervals (e.g. collecting heart rates measured every minute during business hours) [^fn3] by a device such as a smart watch.

In the system pipeline described in the corresponding paper, data is perturbed using Local Differential Privacy, where the data contributor adds noise. Per the pipeline shown below, the user's smart watch identifies salient points in the data streams and then peturbs them with noise, followed by sending the noisy data to the server for reconstruction and storage.

![img](https://lh6.googleusercontent.com/X93uPa9za6kNKEPjejKsQHWMLX7w96gW1yLEj_xERkMiEDrD147G6Fk2buFBtEu2xhMaHahm-5FV8zDwp1RJFaYAywhNlLOBDMXYQzYbdYuSvTWYx8x0XECi7k7WHHMAXprw)


#### Biomedical Dataset Analysis

For the next use case, we will consider handling large data for biomedical applications with differential privacy guarantees. DAMSEN [^fn4] is a system that supports differential privacy guarantees for numerous data analysis tasks and utilizes a effective query optimization engine to achieve high accuracy and low privacy costs.

As demonstrated in the below figure, DAMSEN [^fn4] offers differential privacy for data analysis tasks, such as histograms, cuboids, machine learning algorithms (e.g. linear and logistic regression, potentially generalizable to neural networks), and clustering tasks.

Note: In the context of data analysis tasks apropos queries, histograms do not represent the traditional visualization of the data distribution. Histograms are a special type of query that involves sorting data points into buckets [^fn11]. You can think of such queries as similar to Pandas' groupby() function. A cuboid is an analysis task that involves multiply summary datasets and tables - please see the DAMSEN paper [^fn4] for detailed examples.

TODO: Picture

TODO: Idea on Visualization

An interesting note is that DAMSEN incorporates a compressive mechanism, which is useful for minimizing the amount of noise needed for DP: 

> *“Instead of adding noise to the original data, CM first encodes the data as in compressive sensing; then, CM adds noise to the encoded data, decodes the result as in compressive sensing, and publishes it. Because the transformed data are highly compressed, they require much less noise to achieve differential privacy.”* [^fn5]

#### Analyzing Electronic Health Records

For this use case, we consider DP-perturbed histograms with Homomorphic Encryption [^fn10]. The system proposed in the paper [^fn10] is depicted in the figure below:

TODO: Figure

The concept of the proposed framework is depicted in the below figure. We can see the parts of the framework required for the homomorphic encryption components for key dissemination and the secure histogram generation. In terms of the DP part of the framework: Encrypted Laplace noises are added to the count of each bin of the histogram, where the sensitive of histogram computation is 1
The histograms can be used to train models
Privacy budget needs to be carefully chosen
Security models prevents against various leakages (please see paper for more details)
  



**Differential Privacy References**

[^fn1]: [[1\] Machine learning and genomics: precision medicine versus patient privacy](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2017.0350?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed&)

[^fn2]: [[2\] Emerging technologies towards enhancing privacy in genomic data sharing](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1741-0)

[^fn3]: [[3\] Privacy-preserving aggregation of personal health data streams](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207639)

[^fn4]: [[4\] Demonstration of Damson: Differential Privacy for Analysis of Large Data ](http://differentialprivacy.weebly.com/uploads/9/8/6/2/9862052/pid2574139.pdf)

[^fn5]: [[5\] Compressive Mechanism](https://differentialprivacy.weebly.com/compressive-mechanism.html) 

[^fn6]: [[6\] Project PrivTree: Blurring your “where” for location privacy](https://www.microsoft.com/en-us/research/blog/project-privtree-blurring-location-privacy/)

[^fn7]: [[7\] A History of Census Privacy Protections](https://www.census.gov/library/visualizations/2019/comm/history-privacy-protection.html)

[^fn8]: [[8\] Protecting the Confidentiality of America’s Statistics: Adopting Modern Disclosure Avoidance Methods at the Census Bureau ](https://www.census.gov/newsroom/blogs/research-matters/2018/08/protecting_the_confi.html)

[^fn9]: [9\] [Towards Practical Differential Privacy for SQL Queries](https://arxiv.org/pdf/1706.09479.pdf)

[^fn10]: [10\] [Privacy-preserving biomedical data dissemination via a hybrid approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371369/pdf/2977168.pdf)

[^fn11]: [11\] [Making Histogram Frequency Distributions in SQL](http://www.silota.com/docs/recipes/sql-histogram-summary-frequency-distribution.html)

Additional resource: [Differential privacy: its technological prescriptive using big data](https://link.springer.com/content/pdf/10.1186/s40537-018-0124-9.pdf)

**Differential Privacy Code Repositories**

- [Uber SQL Differential Privacy](https://github.com/uber-archive/sql-differential-privacy)
- [TensorFlow - Differential Privacy](https://blog.tensorflow.org/2019/03/introducing-tensorflow-privacy-learning.html?m=1)
- [Google’s C++ Differential Privacy library](https://github.com/google/differential-privacy)
- [OpenMined Differential Privacy](https://blog.openmined.org/making-algorithms-private/)

Stay posted for a future blog post on use cases for federated learning!

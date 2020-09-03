
# Final Report | Capstone Project – Car Crash Severity in Seattle

### Data

Upon further research for collision data, I obtained a similar data from Seattle GeoData. The data is an open data from the Seattle government in this [link](https://data-seattlecitygis.opendata.arcgis.com/datasets/5b5c745e0f1f48e7a53acec63a0022ab_0?geometry=-124.788%2C47.371%2C-119.732%2C48.018). Data is collected from the year 2004 until present time. The data includes all types of collisions.

The data consists of 40 independent variables as potential features and 221,144 collection of accidents. The dependent variable, **SEVERITYCODE**, contains numbers that correspond to different levels of severity caused by an accident from 0 to 3.

Severity codes are as follows:
    * 3: Fatality — High Probability
    * 2b: Serious Injury — Mild Probability  
    * 2: Injury — Low Probability
    * 1: Property Damage — Very Low Probability
    * 0: Unknown — Little to No Probability

Data needs to be preprocessed, such as removal/fill in null values and encoding of features.

![data.png](/img/data.png)

Labels are uneven, hence it k-cross validation and F1 measure are used.

![classes.png](/img/classes.png)

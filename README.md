# ML Based Fruit Mold Detection

## Problem Statement
### Global Hunger

If food loss and waste were a country, it would be the third-highest greenhouse-gas (GHG) emitting nation behind the US and China. Food waste that ends up in landfills releases methane (CH4), a GHG with a global warming potential approximately 28–36 times higher than carbon dioxide (CO2). The resources required to produce uneaten food have an estimated carbon footprint of 3.3bn tonnes of CO2.

### Ailments from eating bacteria infested fruits.
Fruit is an essential and very complete food, indispensable in a balanced diet. But a very prevalent and harmful problem in fruits is the growth of fungi and mold, in their growth, harvesting, transport and distribution. These fungi or mold, not only accelerate spoilage, but can also generate mycotoxins in fruits, causing allergies, asthma and infections in humans.
In a study published in the International Journal of Food Microbiology (2005), 251 samples of fresh fruit, including several varieties of grapes, strawberries, blueberries, raspberries, blackberries and citrus fruits, were analysed. These were disinfected, incubated at room temperature for up to 14 days without supplementary media, and then examined for mold growth. The results were alarming: fungal contamination levels ranged from 9 to 80%. For example, 35% of the grape samples tested were contaminated and supported fungal growth; while 83% of the citrus samples showed fungal growth at levels ranging from 25% to 100% of the fruit tested.
This pathogen growth affects directly the fruit’s shelf life, producing large losses and contributing to food waste that has reached exorbitant levels.

### Practical challenges in manual mold detection
Also, Sorting fruits for mold manually can be a challenging task due to various practical problems -
•	Subjectivity: Different individuals may have varying levels of sensitivity in detecting mold visually, leading to inconsistencies in the sorting process.
•	Labor Intensive: Manual sorting is labour-intensive and time-consuming. It may require a significant workforce for large quantities of fruits, leading to increased operational costs.
•	Human Error: Humans may overlook subtle signs of mold or may mistakenly identify non-moldy areas as contaminated, leading to inaccurate sorting.
•	Speed and Efficiency: Manual sorting may not be as fast and efficient as automated methods. As a result, the process may not keep up with the pace of production, causing delays.
•	Fatigue: Prolonged manual sorting can lead to worker fatigue, potentially impacting the accuracy of mold detection as attention levels decrease.
•	Inconsistency: The sorting criteria may vary among individuals, leading to inconsistencies in the identification of mold. Standardization becomes challenging in a manual sorting setup.
•	Limited Scale: Manual sorting may not be feasible for large-scale fruit production due to the limitations of human capacity and speed.
•	Health and Safety Concerns: Handling moldy fruits manually can pose health risks to workers, especially if proper safety measures are not in place. Mold spores may cause respiratory issues.
•	Training Requirements: Workers need proper training to identify different types and stages of mold accurately. This training adds to the overall cost and time invested in the sorting process.
•	Quality Control Challenges: Maintaining consistent quality control is difficult in a manual sorting process, as it heavily relies on the capabilities and vigilance of individual workers.
•	Seasonal Variations: The prevalence of mold may vary seasonally, making it challenging for manual sorters to adapt to changing conditions.
•	Waste Management: Disposal of moldy fruits requires proper waste management procedures to prevent contamination and the spread of mold in the sorting facility.

## My solution & its benefits
My project aims to address this issue by leveraging machine learning algorithms to accurately identify and classify moldy fruits based on visual cues.

The software I wrote grounds up utilizes a diverse dataset of fruit images, incorporating various types and sizes. Through a comprehensive training process, the AI model learns to recognize patterns and features associated with mold growth.

The software was tested rigorously on a separate set of fruit images, evaluating the accuracy, precision, and recall of the mold detection algorithm. Apart from the fruits already build in, the software can easily accommodate new fruit varieties and has potential applicability across various contexts in the food industry.

### Solution Implementation
This program is written in python and separated into 4 distinct processes: reading raw data, processing, separation into training and testing data, and model creation. The raw data comes from Mendeley Data and is separated into 16 classes: rotten and fresh for each type of fruit. This structure is the basis for the program, as it includes the image, and its class can be derived from its image path. The program goes through this folder and processes the images to be grayscale and have set dimensions. Next, it shuffles the list of images and gives a 70% weight to training data and a 30% weight to testing data. According to these weights, 70% of the refined images should be considered training data, while 30% of the images should be testing data. Lastly, it creates a Machine Learning model that uses this data and the RELU activation function. This model continues training for 75 iterations, or epochs. It also randomly drops out some connections to prevent it from making false conclusions. Finally, it saves this model to a file so it can be used for future reference.
There were many challenges I had while creating this program, most of which were practical and regarded integration with other modules. These problems were resolved with open source code help from stack overflow and mentors who guided me through the entire process.

## Productization 
A modern fruit sorting machine can be equipped with advanced technology features cameras positioned on four sides, as well as the top and bottom, to ensure comprehensive and accurate fruit inspection. As fruits roll in on a transparent conveyor belt, the cameras capture high-resolution images of the fruits from multiple angles simultaneously. This multi-view imaging system allows for a thorough examination of the entire surface of each fruit.
The transparent conveyor belt serves a dual purpose. Firstly, it enables efficient movement of fruits through the sorting process. Secondly, it facilitates an unobstructed view for the cameras, ensuring that no part of the fruit is hidden during inspection.
The cameras are strategically placed to capture detailed information about the size, color, shape, and surface characteristics of the fruits. Advanced image processing algorithms analyze these images in real-time to detect defects, blemishes, or signs of mold on the fruits.
The machine's software is programmed with predefined sorting criteria, enabling it to categorize fruits based on quality and other attributes. Once a fruit is identified as having mold or other defects, the sorting machine employs a mechanism to divert it to a separate outlet, preventing it from entering the final product stream.
The entire sorting process is automated, significantly increasing efficiency and throughput compared to manual sorting. Additionally, the machine provides detailed data and statistics on the quality and quantity of sorted fruits, facilitating better quality control and production management.

## Conclusion 
The significance of this project lies in its contribution to food safety measures, offering a non-invasive and efficient tool for the early detection of mold in fruits. By automating the inspection process, the software has the potential to reduce food waste, improve consumer health outcomes, and enhance overall quality control in the food supply chain.


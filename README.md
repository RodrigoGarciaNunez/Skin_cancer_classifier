This project implemenst a machine learning model to classify diffrent diagnosis of skin lessions.
The dataset used for this project, HAM-10000, is property of the ISIC-Archive.

The machine learning model implemented here is a CNN, built using tensorflow-keras functional api.

The model obtained from trainning is used in web app implemented using google cloud platform CLI, and FLASK.
Here you can watch the project running: https://skincancerapp-935771581787.northamerica-south1.run.app

The model is able to classify between 8 kinds of skin lessions:
- actinic keratosis
<img src="skin_cancer_app/static/images/ISIC_0024306.jpg" width="300"/>

![](skin_cancer_app/static/images/ISIC_0024306.jpg)
- basal cell carcinoma
<img src="skin_cancer_app/static/images/ISIC_0024310.jpg" width="300"/>

![](skin_cancer_app/static/images/ISIC_0024310.jpg)
- dermatofibroma
<img src="./skin_cancer_app/static/images/ISIC_0024318.jpg" width="300"/>

![](skin_cancer_app/static/images/ISIC_0024318.jpg)
- melanoma
<img src="./skin_cancer_app/static/images/ISIC_0024324.jpg" width="300"/>

![](skin_cancer_app/static/images/ISIC_0024324.jpg)
- nevus
<img src="./skin_cancer_app/static/images/ISIC_0024329.jpg" width="300"/>

![](skin_cancer_app/static/images/ISIC_0024329.jpg)
- pigmented benign keratosis
<img src="./skin_cancer_app/static/images/ISIC_0024331.jpg" width="300"/>

![](skin_cancer_app/static/images/ISIC_0024331.jpg)
- squamus cell carcinoma
<img src="./skin_cancer_app/static/images/ISIC_0024370.jpg" width="300"/>

![](skin_cancer_app/static/images/ISIC_0024370.jpg)
- vascular lesion



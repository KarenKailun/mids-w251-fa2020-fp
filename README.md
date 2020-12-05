<!-----
NEW: Check the "Suppress top comment" option to remove this info from the output.

Conversion time: 1.828 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β29
* Fri Dec 04 2020 22:12:59 GMT-0800 (PST)
* Source doc: Deep Breath white paper

WARNING:
Inline drawings not supported: look for ">>>>>  gd2md-html alert:  inline drawings..." in output.

* Tables are currently converted to HTML tables.
* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 3.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>


Deep Breath: Contactless respiratory rate detection from video

Justin Trobec, Yekun Wang, Karen Wong

W251-001 | Fall 2020

[Presentation](https://github.com/jtrobec/mids-w251-fa2020-fp/blob/main/Breath%20Rate%20Diagrams.pdf)

**Abstract: **

In this paper, we present a method for estimating patient respiratory rate from a contactless, video-based recording, which captures the patient’s breathing pattern in the upright position. Such technology can potentially greatly improve the efficiency for monitoring patients’ vitals, at the same time reducing the risks for unwarranted infectious exposure of the healthcare provider. In a clinical setting, the healthcare provider can monitor the patients’ breathing pattern from a hand-held tablet. 

 

In order to estimate the patient’s respiratory rate, our current methodology utilizes pose detection to infer keypoints on the patient’s body and track their movements through time. Eight keypoint features and three derived distances features serve as inputs to an ensemble model, which combines both a CNN and a frequency estimation via fast Fourier transformation. The output from the model is a numeric value for estimated respiratory rate. Our best model produced the validation MAE of 5.49.

**Introduction**

The COVID-19 pandemic has highlighted the need for video-based medical monitoring methods that reduce risk of infectious exposure for healthcare providers. Healthcare facilities have increased their use of telemedicine, to avoid unnecessary infectious exposures for staff and for patients (Koonin 2020). The ability to monitor COVID-19 patients at home can alleviate burden on the healthcare system, particularly during resource-constrained periods of pandemic surge. Within the healthcare setting, contactless monitoring can also reduce waste of disposable monitoring equipment and conserve valuable personal protective equipment (PPE). 

	Disturbances in respiratory rate (breaths per minute) can be an early sign of clinical deterioration; increases can indicate respiratory distress or worsening metabolic status. However, the traditional method of manual counting is often recorded inaccurately — in practice, staff often estimate a respiratory rate without timing the breaths per minute. Monitoring equipment can be uncomfortable, particularly for people in respiratory distress, children, or patients with altered mental status. 

	Contactless respiratory rate monitoring has been developed previously; however, several of these rely on equipment not readily available to most patients, such as infrared or thermal cameras (Pereira 2015), or wifi signal detectors (Ravichandran 2015). Using a simple webcam would make this technology more widely available. Additionally, prior models that have been trained to detect respiratory rate have often used wearable sensors to provide training data (Chu 2019). 

	We aimed to develop a contactless, video-based monitoring system for respiratory rate that offered low latency and privacy preservation without need for extra equipment on computing power on the patient side, and without need for wearable sensors to provide training data. 

**Methods**

Our model uses front-facing, upper body video of a person to mimic a telemedicine encounter, where patients are likely to be seated and facing their computer, to predict a respiratory rate. Face and upper body keypoints are detected in real time from the video and used as inputs to a respiratory rate prediction model. 

_Pipeline_

Describe the whole pipeline -- patient’s edge device to cloud to provider. Include description of web UI. 



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")


_Training data_

We recorded 259 samples of 15-second video clips annotated with the breathing rate ranging from 8 to 41 breaths/minute. Videos included three different adults facing the camera with the face and upper body included in the frame. We applied an 85-15 training-test split to the data, and during training used a 0.1 validation split. 

_Person and keypoint detection task_

To detect a person and identify keypoints simultaneously from video, we used ResNet18 (He 2016), a convolutional neural network model pre-trained for pose detection using data from MSCOCO (Lin 2014; trt_pose available at [https://github.com/NVIDIA-AI-IOT/trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)). See [pose detection notebook](https://github.com/jtrobec/mids-w251-fa2020-fp/blob/main/posedetect_fromvideo.ipynb). 

_Feature selection, engineering, and transformation_

We selected as inputs to the respiratory rate detection model the movement of the following y-coordinate keypoints: bilateral eyes, ears, and shoulders; nose; and base of the neck. In addition, we calculated the Euclidean distances over time from right ear to shoulder, left ear to shoulder, and nose to neck as inputs to the model. Y-coordinates of keypoints and the distances were normalized according to the training data. Missing values, which represent keypoints not being detected because they were out of frame or because the keypoint detection model failed to detect them, were zero-filled after normalization. In addition, a column of frame index is also included as a feature input to the model. Together, there are 12 features in total (8 keypoints, 3 derived distance features, and 1 frame index). See `process_keypoints` [notebook](https://github.com/jtrobec/mids-w251-fa2020-fp/blob/main/process_keypoints.ipynb). 

_Respiratory rate detection model_

The main respiratory rate detection model was an ensemble of fast Fourier transform frequency estimation with a CNN. The input to the model were the values over time of selected upper body keypoints and distances. The fast Fourier transformation was applied to each feature to estimate its frequency along the time dimension, which effectively produced a vector of length 12. The input to the CNN has the shape of [440 x 12 x 1], for a 15-second long clip recorded at 30 fps. For the CNN, we used filters the width of the input data and approximately 1 second long. The output from the CNN model was flatten and resulted in a vector of length 3296. Subsequently, the CNN output and fast Fourier output were concatenated into a vector of length 3308. This vector was then treated as the input vector to a feedforward neural network, which consisted of two fully connected layers and a dropout layer. Finally, the network outputs a single numeric prediction for the breathing rate of the input video clip. See 



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)

We chose MSE (mean squared error) as the loss function was mean squared error and Adam optimizer. Training occurred on an edge device (NVIDIA Jetson Xavier NX or AGX Xavier). 

	We also attempted variations on the model architecture. A 3D CNN was used for a different data structure where each keypoint was stacked depthwise to represent different channels. Additionally, we also tried a time series approach for calculating autocorrelation and counting the periodicity to predict breathing rate of each keypoint. Based on the predicted breathing rate of each keypoint, we also tried other machine learning approaches such as linear regression and tree-based methods to validate the relative importance of each keypoint.

**Results**

Given a well-behaved periodic function, either the frequency estimation from a fast Fourier transformation or counting the periodicity in the autocorrelation of the timeseries should produce decent results. Empirically, we saw that the keypoints representing the shoulders tend to produce the cleanest periodic form. However, given the various camera angles and patients’ movement during the recording, it’s possible that other keypoints may produce more consistent estimates. Therefore, the goal of training a machine learning model becomes picking the most relevant keypoints for each sample.

The best performing model was the ensemble of fast Fourier transform with a CNN. The additional models were unable to achieve meaningful results. 


<table>
  <tr>
   <td><strong>Model</strong>
   </td>
   <td><strong>Training MAE</strong>
   </td>
   <td><strong>Validation MAE</strong>
   </td>
  </tr>
  <tr>
   <td>Frequency Estimation + Heuristics
   </td>
   <td>n/a
   </td>
   <td>10.89
   </td>
  </tr>
  <tr>
   <td>Frequency Estimation + LR
   </td>
   <td>8.44
   </td>
   <td>7.91
   </td>
  </tr>
  <tr>
   <td>FFT + Feedforward Network
   </td>
   <td>6.31
   </td>
   <td>9.50
   </td>
  </tr>
  <tr>
   <td>CNN
   </td>
   <td>3.42
   </td>
   <td>7.37
   </td>
  </tr>
  <tr>
   <td>Ensemble: FFT FNN + CNN
   </td>
   <td>2.79
   </td>
   <td>5.49
   </td>
  </tr>
</table>


Overall, predictions for each 15-second clip correlated well with the true respiratory rate, although there was some variability between multiple clips from the same longer video. 



<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")


_Future Work_

Though our current methodology shows promising results, there is still lots of room for improvements. To improve data quality, future research can consider using an infrared and/or thermal camera. To improve the data collection efficiency, instead of truncating input video into 15-second clips, future research could consider a moving-window approach to produce a stream of 15-second clips. In order to capture better keypoints, future research can also consider other optical flow methods for tracking an object through time.

**Conclusion**

This represents a successful proof of concept for real-time respiratory rate detection from contactless video-based monitoring. We developed a model that predicts respiratory rate from the upper body keypoint motion; performing the keypoint detection preserves patient privacy and also allows the respiratory rate detection model to be lightweight. We developed a working pipeline that ingests patient-side video, performs inference on the patient’s edge device, transmits predictions to the cloud, and allows healthcare providers to view the estimated respiratory rate with low latency. 

	Our model is currently limited to detection of respiratory rate in artificial conditions where the patient is facing the camera, not moving or talking, with little motion in the background. It currently accommodates only one person in frame at a time. The model currently estimates respiratory rate for 15-second video clips. 

	There are several improvements to this model that would enhance its usefulness for clinical and other applications. Broadening the training examples to include more individuals is important to represent age, sex, and racial/ethnic diversity in patients. The model is currently trained to mimic a telemedicine encounter, but it would be more flexible if it could account for different poses, such as side or top-down views. Much more training data are needed to allow the model to detect respiratory rate while a patient is speaking, eating, or moving. The ability to detect and monitor multiple individuals could be useful in clinical settings (emergency department, intensive care, field hospitals) where multiple patients need to be monitored simultaneously. 

	Beyond respiratory rate detection, our concept could have other video monitoring applications at home and in the clinical setting. Home baby and child monitors could detect life-threatening irregularities in breathing patterns. Sleep studies, which typically require a patient to sleep overnight in a hospital while being connected to uncomfortable monitoring equipment, could instead be done in the patient’s home with a simple webcam. Seizure monitoring could involve video screening in a patient’s home, where a model trained to detect seizure movements could be triggered to record video or alert caregivers when a seizure begins. By learning the location and characteristics of seizure movements, a model can even suggest the anatomical location of the epileptic focus in the brain. Our proof of concept suggests that these additional applications, which traditionally incur large healthcare costs, can succeed with minimal cost and equipment, even in resource-limited parts of the world where more expensive monitoring technologies and trained healthcare providers are not available. 

**References**

Chu, M., Nguyen, T., Pandey, V. et al. Respiration rate and volume measurements using wearable strain sensors. npj Digital Med 2, 8 (2019). https://doi.org/10.1038/s41746-019-0083-3

He, K., et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

Koonin, L.M., et al. "Trends in the Use of Telehealth During the Emergence of the COVID-19 Pandemic—United States, January–March 2020." Morbidity and Mortality Weekly Report 69.43 (2020): 1595.

Lin, T., et al. "Microsoft COCO: Common objects in context. arXiv 2014." arXiv preprint arXiv:1405.0312 (2014).

Pereira C.B., Yu X., Blazek V., et al. Robust remote monitoring of breathing function by using infrared thermography. Annu Int Conf IEEE Eng Med Biol Soc. 2015;2015:4250-3. doi: 10.1109/EMBC.2015.7319333. PMID: 26737233.

Ravichandran, R., Saba, E., Chen, K., et al. "WiBreathe: Estimating respiration rate using wireless signals in natural settings in the home," 2015 IEEE International Conference on Pervasive Computing and Communications (PerCom), St. Louis, MO, 2015, pp. 131-139, doi: 10.1109/PERCOM.2015.7146519.

# Measuring Infant Respiratory Rate via Computer Vision
MIDS W251 Fall 2020 Final Project - Karen Wong, Yekun Wang, Justin Trobec

## Overview

We seek to build a system that can track the respiratory rate of a sleeping infant through unobtrusive means, such as a low-cost infrared (IR) camera. This is of potential interest to health-care professionals, parents of children with respiratory issues, and parents who are reassured by having more insights into their child's health than a typical monitor can provide.

## Proposed Architecture

### Edge

We will use NVidia Xavier NX devices with a few different camera types: integrated standard and IR cameras, and external USB cameras. These will run a deep-learning model that observes key visual features of a sleeping child, and transforms them into a periodic measurement of respiratory rate in breaths per minute. The device will transmit measurements to a cloud device, along with clips of significant sleep events where breath measurements are interrupted.

### Cloud

We will leverage cloud technologies for two main purposes:

1) storing and surfacing respiratory rate data.
2) training and customization (transfer learning) of models.

For 1, we envision exposing an API that can be queried by a device like a phone in order to check current respiratory rates, or provide notifications when there are interruptions to respiration or rates cross defined thresholds.

For 2, there is the possibility that we may need to build new or customize existing models for tracking key points or edges.

## Evaluation

Our system can be described as a function that takes as input a 1min video, and outputs either a respiratory rate, or an indicator that we cannot determine the respiratory rate from the image. Our accuracy is the percent of correct outputs from this function over an annotated training set.

## References

Infant pose estimation and infant movement-based assessment
https://github.com/cchamber/Infant_movement_assessment

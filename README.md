# Automated-Mineralogy
ML project to predict mineral maps from cross-polarized (XPL) and plain-polarized (PPL) light thin section photographs.

# Introduction
Understanding the mineral composition of rocks is a very important topic for nearly all fields of geology. Thin sections (0.03 mm thick slices of rock) are the primary way in which researchers are able to get detailed information regarding the composition, metamorphism, and paragenetic history of a rock sample. By viewing thin sections under different light polarizations, trained mineralogists can manually identify which grains are which minearls. 

<figure>
  <p align="center"> <img src='/Images/ppltest.jpg' width="100" height="200"> 
  <figcaption>Example of plain polarized light thin section</figcaption> </p>
</figure>

<figure>
  <p align="center"> <img src='/Images/xpltest.png' width="100" height="200"> 
  <figcaption>Example of plain polarized light thin section</figcaption> </p>
</figure>

Advanced techniques include using Energy Dispersive X-ray (EDX) and backscattered electron (BSE) signals to identify minerals based on the energy emitted back from the thin sections. These signals are then compared to entries in a mineral library of known samples to determine mineral concentrations, element distributions, and mineral textural properties ([link](https://www.sgs.com/en/campaigns/tima-x-automated-mineralogy-system#:~:text=How%20It%20Works,entries%20in%20a%20mineral%20library.)). Although very accurate, this process is time consuming and costly as it requires samples be be mailed off to a lab with the appropriate instruments and software.

# Project Goal
The goal of this project is to take regular images from thin sections taken under plain polarized light (PPL) and cross-polarized light (XPL) and re-produce the type of mineralogy maps you would get from more advanced analysis techniques.

---
title: "TA: Optimization Models and Methods"
collection: teaching
type: "Graduate Course"
description: "TA for IEOR E4004. Topics included linear, nonlinear, integer, and dynamic programming."
permalink: /teaching/2023-spring-optimization
venue: "Columbia University, Dept. of IEOR"
date: 2023-01-01
location: "New York, NY"
importance: 1
---

I served as the Teaching Assistant for Graduate Optimization Models and Methods (IEOR E4004). The course covers linear programming, the simplex method, duality, and nonlinear, integer, and dynamic programming.

My duties included:
*   Grading homework and the course project, providing detailed feedback.
*   Revising and improving official solutions for assignments.
*   Enhancing the final project, "Moving Object Detection," by creating a more robust and user-friendly coding template.

## Final Project Enhancement: Moving Object Detection

I think this is a very interesting project with inspiring questions. The description of this project is as below:
## Description

Suppose we would like to extract moving objects from a video. We encode the video as a data matrix $$M ∈ \mathbb{R}^{m×n}$$ where each column corresponds to a video frame, and each row corresponds to a pixel. If all the columns are the same, then there is no movement in the video and the data matrix $M$ has rank one. If one or more objects are moving, then they can be can be viewed as noise $$ε ∈ \mathbb{R}^{m×n}$$ that is being added to a rank one matrix $$uv^T$$ (the fixed background) for some $$u ∈ \mathbb{R}^m$$ and $$v ∈ \mathbb{R}^n$$. In other words, we have $$M = uv^T + ε$$. One can thus try to recover the background (and hence the moving objects) by finding a closest rank one matrix $$xy^T$$ to the data matrix $M$, as follows:

$$\inf_{(x,y)\in\mathbb{R}^m\times\mathbb{R}^n}\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^n(x_iy_j-M_{ij})^2$$

## Selected Questions
1. Notice that we do not assume that all the elements of the vector $v$ are equal to one another. What is the benefit of allowing them to be different from one another?
1. Determine the update rule of the stochastic gradient descent method applied to the objective function.
1. Is the objective function convex? Show that all local minima are global minima when $$ε = 0$$. Comment on the empirical success of the approach.

## Enhancements for the coding part

I have noticed that some students are having trouble with the current code template, for instance, some students correctly derived the SGD update rule while left the block empty where they are supposed to implement the SGD method. 

And when I was trying to do the project, I noticed that some operations require a lot of RAM, and the generated video files are larger than the original video, which could be an issue if the students only have limited storage. 

Therefore, I have created a [colab version template](https://drive.google.com/file/d/1mkY8H5v3MTXh2UEaNygD6LyHrq61mJiv/view?usp=sharing)(LionMail required) for the project that can be viewed by LionMail. Students can make a copy of the colab notebook and implement the algorithm. The features are listed below: 

Features: 

* Colab notebook
  * Colab solves the environment issues, and provides enough storage space
* `SGD` function for students to implement
  * The function is put at the beginning of the notebook, so that students could easily figure out what they are supposed to implement
* Supports directly load video from Youtube
  * It might be easier to have the option to load the video of interest from Youtube
* A simple moving circle video is provided by default
  * I have uploaded a 3s moving circle video, students can use this video to test their implementation before applying the algorithm to larger video
  * ![circle](/assets/img/circle.png)
* Clearer hypermarameters
  * ![param-mod](/assets/img/param-mod.png)
  * Colab supports form field, which could be clearer for defining hyper parameters
* Use `numpy` for batch update, instead of `for loop`
  * The optimization time could be greatly reduced if the `batch_size` is large. Below is a comparison of `numpy` method and `for loop` method with `batch_size=32`
  * `numpy`: 1min ![np](/assets/img/np.png)
  * `for loop`: 13min![for](/assets/img/for.png)
* Convert type `float64` to `uint8` before $xy^T$ to prevent out-of-memory
  * For the video of 56Kb (simple moving circle), the `uint8` `numpy.ndarray` is ~60Mb, `float64` is ~480Mb. It could easily run out og memory if not converting type before the multiplication. ![size](/assets/img/size.png)
  * The quality of the result video is similar:
    * convert to `uint8` before multiplication:![uint8_before](/assets/img/uint8_before.png)
    * convert to `uint8` after multiplication:![uint8_after](/assets/img/uint8_after.png)
* Changed variable name `L`, `S` to `estimate_background` and `noise`
* Used `ffmpeg` to reduce the size of output video
  * The result video files are much larger than the input video. For instance, the 56Kb video would result in a `background.mp4` of size ~750Kb, and a `moving_object.mp4` of size ~4.6Mb. While using `ffmpeg` could reduce the size to ~160Kb and ~830Kb, respectively.
* Added an embedded video display feature
  * It could be easier for students to display and check their result video on Colab
  * ![display](/assets/img/display.png)



Limitations:

* Video larger than RAM
  * The whole video is read in RAM for subsequent operations, so if the video is larger than RAM, the session will be crashed
* Large result video
  * Large result video is not able to be displayed by current embedded video display feature, the students might have to download the video and check, which could consume longer time since the download speed of Colab is not fast.


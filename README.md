# sabertracker
## A computer vision project to detect lightsabers in video


[<img src="https://www.jointhelight.ludosport.net/wp-content/uploads/2018/04/logo_ex2.png">](https://www.youtube.com/watch?v=6_S2_C582lw "What is LudoSport?!")

### Why?
During my PhD, I learned alot about computer vision and how to apply image processing techniques to microscopy data. Now I want to apply this knowledge to other areas.

I've been practising LudoSport here in Ireland since early 2020 and since one of our practitioners records our sparring sessions to help us identify weaknesses and mistakes to improve our teqchnique, I started thinking that I could get a computer to automatically analyse these data and give us more information.

One critique has been that we too often fall back to guard stance, which has a very stationary position that is amenable to detection by monitoring the movement of the blade. <em>This is the current aim of this tracker, but it may become more sophisticated in time</em>

### What?
This repo documents my process from inception to fully-fledged lightsaber tracker. I'm trying to stay away from AI methods like Yolo models, which perform really well and fast if you can take the time to annotate lots of data and have a decent GPU. However, they are not very fast on regular CPUs and I intend to host the app on something free e.g. Streamlit Cloud, so it's important that this program runs fast on a regular CPU. For reference, I am developing on a laptop with 8GB RAM, an Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz CPU, and integrated GPU.

<strong>Currently</strong>, I've nearly finished optimizing the centroid detection algorithm to a satisfactory FPS. <strong>Next</strong> I will implement a centroid tracker to follow lightsabers in the videos.

### TODO
* Implement centroid tracker
* Extract metrics from tracked sabers
* Create and deploy web app
